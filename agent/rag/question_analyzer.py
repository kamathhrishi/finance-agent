#!/usr/bin/env python3
"""
Question Analyzer for the RAG system.

This module handles question analysis, validation, and preprocessing for the RAG system.
It uses AI models to analyze questions and extract relevant information like tickers,
quarter references, and question types.
"""

import os
import json
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
import openai

# Import local modules
from .config import Config
from app.schemas.rag import QuestionAnalysisResult
from .conversation_memory import ConversationMemory
from .rag_utils import parse_json_with_repair
from agent.prompts import (
    TICKER_REPHRASING_SYSTEM_PROMPT,
    get_ticker_rephrasing_prompt
)

# Import Logfire for observability (optional)
try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None

# Configure logging
logger = logging.getLogger(__name__)
rag_logger = logging.getLogger('rag_system')


# ═════════════════════════════════════════════════════════════════════
# STAGE 1: INITIALIZATION & SETUP
# ═════════════════════════════════════════════════════════════════════

class QuestionAnalyzer:
    """Analyzes and preprocesses questions for the RAG system."""
    
    def __init__(self, openai_api_key: Optional[str] = None, config: Config = None, database_manager = None):
        """Initialize the question analyzer."""
        # Use Cerebras API key
        cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
        if not cerebras_api_key:
            raise ValueError("CEREBRAS_API_KEY is required for question analysis")
        
        self.cerebras_api_key = cerebras_api_key
        self.config = config or Config()  # Use provided config or create new one
        self.database_manager = database_manager  # Store database_manager for company-specific quarter queries
        
        # Initialize Cerebras client
        try:
            from cerebras.cloud.sdk import Cerebras
            self.client = Cerebras(api_key=cerebras_api_key)
            self.cerebras_available = True
            logger.info(f"✅ QuestionAnalyzer initialized successfully with Cerebras (model: {self.config.get('cerebras_model')})")
        except ImportError:
            logger.error("❌ Cerebras SDK not installed. Run: pip install cerebras-cloud-sdk")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize Cerebras client: {e}")
            raise
        
        # Initialize conversation memory
        self.conversation_memory = ConversationMemory()

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 2: QUESTION ANALYSIS (LLM-based Analysis)
    # ═════════════════════════════════════════════════════════════════════

    async def analyze_question(self, question: str, conversation_id: str = None, db_connection = None) -> Dict[str, Any]:
        """Analyze a question and determine the appropriate data source (earnings transcripts, 10-K filings, or news)."""
        rag_logger.info(f"🔍 Starting question analysis for: '{question}'")

        # Retry configuration - increased for better JSON parsing reliability
        max_retries = 8
        base_delay = 0.5  # seconds
        
        for attempt in range(max_retries):
            try:
                # Get conversation context if conversation_id provided
                conversation_context = ""
                has_conversation_context = False
                if conversation_id:
                    # Get conversation context using the conversation memory system
                    conversation_context = await self.conversation_memory.format_context(conversation_id)
                    
                    if conversation_context:
                        has_conversation_context = True
                        rag_logger.info(f"📚 Conversation context retrieved ({len(conversation_context)} chars)")
                        rag_logger.info(f"📚 Context preview: {conversation_context[:200]}...")
                        conversation_context = f"\n{conversation_context}\n"
                    else:
                        rag_logger.info(f"📚 No conversation context found for conversation_id: {conversation_id}")
                
                # Get comprehensive quarter context for LLM
                quarter_context = self.config.get_quarter_context_for_llm()
                
                # Build context-aware instructions for ticker extraction
                # Note: Semantic grounding rule is defined once in the CRITICAL RULE section below
                if has_conversation_context:
                    ticker_instructions = """1. Extract company tickers from $TICKER format (e.g., $AAPL -> AAPL)
   - **CRITICAL**: The question uses references to previous conversation. YOU MUST extract ALL relevant company tickers from the conversation context above
   - Look through the ENTIRE conversation context to find ALL ticker symbols that were mentioned in previous User or Assistant messages
   - If the question says "all companies mentioned above", "those companies", "these companies" OR uses pronouns like "their", "they", "it", extract ALL tickers found in the conversation, not just one
   - For contextual references, the conversation history is your primary source of company information
   - Apply SEMANTIC GROUNDING rule (see below) to rephrased_question"""
                else:
                    ticker_instructions = """1. Extract company tickers from $TICKER format (e.g., $AAPL -> AAPL)
   - Apply SEMANTIC GROUNDING rule (see below) to rephrased_question"""
                
                # Create analysis prompt following JSON output best practices
                analysis_prompt = f"""Analyze this financial/business question and determine the appropriate data source. Respond with valid JSON only.

QUESTION: "{question}"

{quarter_context}

{conversation_context}

INSTRUCTIONS:
{ticker_instructions}
2. **SEMANTIC DATA SOURCE ROUTING** - Analyze the INTENT of the question to determine the best data source:

   **Think about WHAT TYPE OF INFORMATION would best answer this question:**

   A) **10-K SEC Filings** (data_source="10k", needs_10k=true) - Best for:
      - Annual/full-year financial data, comprehensive financial statements
      - Balance sheets, income statements, cash flow statements, stockholders equity
      - Executive compensation, CEO pay, salary, stock awards, bonuses
      - Risk factors, legal proceedings, regulatory matters
      - Detailed business descriptions, segment breakdowns
      - Audited financial figures, official SEC disclosures
      - Historical trends spanning multiple years
      - Total assets, liabilities, debt levels, capital structure

   B) **Earnings Transcripts** (data_source="earnings_transcripts") - Best for:
      - Quarterly performance discussions, recent quarter results
      - Management commentary, executive statements, tone/sentiment
      - Forward guidance, outlook, projections for upcoming quarters
      - Analyst Q&A, investor concerns, management responses
      - Product launches, strategic initiatives discussed in calls
      - Quarter-over-quarter comparisons, sequential trends

   C) **Latest News** (data_source="latest_news", needs_latest_news=true) - Best for:
      - Very recent events (last few days/weeks)
      - Breaking developments, announcements, market reactions
      - Current market sentiment, stock movements
      - Recent partnerships, acquisitions, leadership changes
      - Events that may not yet be in financial filings

   D) **Hybrid** (data_source="hybrid") - Best for:
      - Questions explicitly requesting multiple perspectives
      - Comparing official filings with recent developments
      - Comprehensive analysis needing both historical and current data

   **ROUTING DECISION PROCESS:**
   1. What is the user trying to learn? (Intent)
   2. What time period is relevant? (Annual=10K, Quarterly=Transcripts, Recent=News)
   3. What level of detail/formality? (Official/Audited=10K, Commentary=Transcripts, Current=News)
   4. Would combining sources provide a better answer? (Consider hybrid)
5. Detect quarter references:
   - YEAR ONLY (e.g., "2024", "2025"): quarter_context: "multiple", quarter_count: 4, quarter_reference: "YYYY_all" (ALL quarters of that year)
   - "last X quarters" or "past X quarters" → quarter_context: "multiple", quarter_count: X
   - "last few quarters" or "recent quarters" → quarter_context: "multiple", quarter_count: 3
   - "latest quarter", "most recent quarter", "current quarter" → quarter_context: "latest", quarter_reference: "latest"
   - Specific quarter (e.g., "Q1 2024", "2024 Q1") → quarter_context: "specific", quarter_reference: "2024_q1"
   - Otherwise → quarter_context: "latest", quarter_count: null

REQUIRED JSON STRUCTURE:
{{
  "is_valid": true,
  "reason": "Brief explanation",
  "question_type": "specific_company|multiple_companies|general_market|financial_metrics|guidance|challenges|outlook|industry_analysis|executive_leadership|business_strategy|company_info|latest_news|invalid",
  "extracted_ticker": "TICKER or null",
  "extracted_tickers": ["TICKER1", "TICKER2"],
  "rephrased_question": "Semantically grounded version (NO company names, NO time periods, ONLY concepts)",
  "suggested_improvements": ["Suggestion 1"],
  "confidence": 0.95,
  "quarter_reference": "2024_q3 or null",
  "quarter_context": "latest|previous|specific|multiple",
  "quarter_count": 3 or null,
  "data_source": "10k|latest_news|earnings_transcripts|hybrid",
  "needs_latest_news": false,
  "needs_10k": false
}}

**CRITICAL RULE FOR rephrased_question - SEMANTIC GROUNDING**:
- REMOVE company names: Transcripts already filtered by ticker
- REMOVE time periods: Transcripts already filtered by quarter/year
- FOCUS on concepts, metrics, and topics
- Example: "What was Apple's Q4 2024 revenue?" → "revenue and sales performance"
- Example: "Compare Microsoft and Google cloud revenue" → "cloud services revenue and growth"
- Example: "Did Amazon discuss AWS in Q3?" → "cloud computing services discussion and performance"

**CRITICAL: DETECTING INVALID QUESTIONS**

**What we CAN answer (mark is_valid=true):**
We have data from PUBLIC COMPANY earnings call transcripts, 10-K SEC filings, and company news.
Valid questions are about:
- Public company financial performance, revenue, earnings, margins, growth
- What management said in earnings calls (guidance, strategy, commentary)
- 10-K filing data (balance sheets, risk factors, executive compensation)
- Company news and recent developments
- Industry trends discussed by public companies
- Comparing public companies

**If the question is NOT about the above, mark is_valid=false.**
This includes: gibberish, greetings, non-finance topics, things we don't have data for, or questions too vague to answer.

When marking as invalid, provide a helpful reason explaining what we CAN help with and suggest example questions.

EXAMPLES:

QUESTION: "Apple's revenue in 2024"
OUTPUT: {{"is_valid": true, "reason": "Valid earnings question", "question_type": "specific_company", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "rephrased_question": "revenue and sales performance", "suggested_improvements": ["Specify exact quarters"], "confidence": 0.9, "quarter_reference": "2024_all", "quarter_context": "multiple", "quarter_count": 4, "data_source": "earnings_transcripts"}}

QUESTION: "Compare Microsoft and Google cloud revenue for the last 3 years"
OUTPUT: {{"is_valid": true, "reason": "Valid earnings question", "question_type": "multiple_companies", "extracted_ticker": "MSFT", "extracted_tickers": ["MSFT", "GOOGL"], "rephrased_question": "cloud services revenue performance and growth trajectory", "suggested_improvements": ["Specify exact quarters"], "confidence": 0.9, "quarter_reference": null, "quarter_context": "multiple", "quarter_count": 12, "data_source": "earnings_transcripts"}}

QUESTION: "What did Apple say about iPhone sales in their latest quarter?"
OUTPUT: {{"is_valid": true, "reason": "Valid earnings question", "question_type": "specific_company", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "rephrased_question": "iPhone product sales discussion and performance", "suggested_improvements": [], "confidence": 0.95, "quarter_reference": "latest", "quarter_context": "latest", "quarter_count": null, "data_source": "earnings_transcripts"}}

QUESTION: "How has Microsoft's revenue changed over the last 3 quarters?"
OUTPUT: {{"is_valid": true, "reason": "Valid earnings question", "question_type": "specific_company", "extracted_ticker": "MSFT", "extracted_tickers": ["MSFT"], "rephrased_question": "revenue changes and growth trends", "suggested_improvements": [], "confidence": 0.9, "quarter_reference": null, "quarter_context": "multiple", "quarter_count": 3, "data_source": "earnings_transcripts", "needs_latest_news": false}}

QUESTION: "What's the latest news on NVIDIA?"
OUTPUT: {{"is_valid": true, "reason": "Question about latest news", "question_type": "latest_news", "extracted_ticker": "NVDA", "extracted_tickers": ["NVDA"], "rephrased_question": "latest news and current developments", "suggested_improvements": [], "confidence": 0.95, "quarter_reference": null, "quarter_context": "latest", "quarter_count": null, "data_source": "latest_news", "needs_latest_news": true}}

QUESTION: "Find me all latest news on nvidia"
OUTPUT: {{"is_valid": true, "reason": "Question explicitly asking for latest news", "question_type": "latest_news", "extracted_ticker": "NVDA", "extracted_tickers": ["NVDA"], "rephrased_question": "latest news and current information", "suggested_improvements": [], "confidence": 0.95, "quarter_reference": null, "quarter_context": "latest", "quarter_count": null, "data_source": "latest_news", "needs_latest_news": true}}

QUESTION: "What was Tim Cook's compensation in 2023? Find out from the 10k"
OUTPUT: {{"is_valid": true, "reason": "Question about executive compensation from 10-K filing", "question_type": "executive_leadership", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "rephrased_question": "executive compensation and pay details for chief executive officer", "suggested_improvements": [], "confidence": 0.95, "quarter_reference": "2023_all", "quarter_context": "multiple", "quarter_count": 4, "data_source": "10k", "needs_10k": true}}

QUESTION: "Find out Tim cooks compensation from 10k for 2023"
OUTPUT: {{"is_valid": true, "reason": "Question explicitly mentions 10k and asks about executive compensation", "question_type": "executive_leadership", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "rephrased_question": "executive compensation and pay details for chief executive officer", "suggested_improvements": [], "confidence": 0.95, "quarter_reference": "2023_all", "quarter_context": "multiple", "quarter_count": 4, "data_source": "10k", "needs_10k": true}}

QUESTION: "find out Tim cooks compensation in 2023"
OUTPUT: {{"is_valid": true, "reason": "Question about executive compensation - this data is only in 10-K filings, not earnings transcripts", "question_type": "executive_leadership", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "rephrased_question": "executive compensation and pay details for chief executive officer", "suggested_improvements": [], "confidence": 0.95, "quarter_reference": "2023_all", "quarter_context": "multiple", "quarter_count": 4, "data_source": "10k", "needs_10k": true}}

QUESTION: "What was the CEO's salary at Apple in 2023?"
OUTPUT: {{"is_valid": true, "reason": "Question about CEO salary - executive compensation is only in 10-K filings", "question_type": "executive_leadership", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "rephrased_question": "chief executive officer salary and compensation", "suggested_improvements": [], "confidence": 0.95, "quarter_reference": "2023_all", "quarter_context": "multiple", "quarter_count": 4, "data_source": "10k", "needs_10k": true}}

QUESTION: "Show me Apple's balance sheet from their annual report"
OUTPUT: {{"is_valid": true, "reason": "Question about balance sheet from 10-K", "question_type": "financial_metrics", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "rephrased_question": "balance sheet and financial position", "suggested_improvements": [], "confidence": 0.95, "quarter_reference": "latest", "quarter_context": "latest", "quarter_count": null, "data_source": "10k", "needs_10k": true}}

QUESTION: "What are Apple's total assets from their 10-K filing?"
OUTPUT: {{"is_valid": true, "reason": "Question explicitly mentions 10-K and asks about financial statements", "question_type": "financial_metrics", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "rephrased_question": "total assets and balance sheet information", "suggested_improvements": [], "confidence": 0.95, "quarter_reference": "latest", "quarter_context": "latest", "quarter_count": null, "data_source": "10k", "needs_10k": true}}

QUESTION: "Get me Microsoft's risk factors from the 10k"
OUTPUT: {{"is_valid": true, "reason": "Question explicitly mentions 10k and asks about risk factors", "question_type": "company_info", "extracted_ticker": "MSFT", "extracted_tickers": ["MSFT"], "rephrased_question": "risk factors and business risks", "suggested_improvements": [], "confidence": 0.95, "quarter_reference": "latest", "quarter_context": "latest", "quarter_count": null, "data_source": "10k", "needs_10k": true}}

QUESTION: "Show me Google's income statement from their 10-K"
OUTPUT: {{"is_valid": true, "reason": "Question explicitly mentions 10-K and asks about financial statements", "question_type": "financial_metrics", "extracted_ticker": "GOOGL", "extracted_tickers": ["GOOGL"], "rephrased_question": "income statement and financial performance", "suggested_improvements": [], "confidence": 0.95, "quarter_reference": "latest", "quarter_context": "latest", "quarter_count": null, "data_source": "10k", "needs_10k": true}}

QUESTION: "What did Apple say about their revenue in Q4 2024?"
OUTPUT: {{"is_valid": true, "reason": "Question about quarterly earnings discussion", "question_type": "specific_company", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "rephrased_question": "revenue performance and sales results", "suggested_improvements": [], "confidence": 0.95, "quarter_reference": "2024_q4", "quarter_context": "specific", "quarter_count": null, "data_source": "earnings_transcripts", "needs_10k": false}}

QUESTION: "Compare Apple's revenue in Q4 2024 earnings call vs their annual report"
OUTPUT: {{"is_valid": true, "reason": "Hybrid question comparing earnings call with annual report (10-K)", "question_type": "financial_metrics", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "rephrased_question": "revenue comparison and financial performance", "suggested_improvements": [], "confidence": 0.9, "quarter_reference": "2024_q4", "quarter_context": "specific", "quarter_count": null, "data_source": "hybrid", "needs_10k": true}}

QUESTION: "Analyze Google's AI strategy using earnings calls and latest news"
OUTPUT: {{"is_valid": true, "reason": "Hybrid question asking for both earnings data and latest news", "question_type": "business_strategy", "extracted_ticker": "GOOGL", "extracted_tickers": ["GOOGL"], "rephrased_question": "artificial intelligence strategy and initiatives", "suggested_improvements": [], "confidence": 0.9, "quarter_reference": "latest", "quarter_context": "latest", "quarter_count": null, "data_source": "hybrid", "needs_latest_news": true}}

QUESTION: "Compare Microsoft's Q4 2024 revenue with recent news about their products"
OUTPUT: {{"is_valid": true, "reason": "Hybrid question combining specific quarter analysis with recent news", "question_type": "specific_company", "extracted_ticker": "MSFT", "extracted_tickers": ["MSFT"], "rephrased_question": "revenue performance and product developments", "suggested_improvements": [], "confidence": 0.9, "quarter_reference": "2024_q4", "quarter_context": "specific", "quarter_count": null, "data_source": "hybrid", "needs_latest_news": true}}

QUESTION: "wrekashfkjbhkl;ahsnhbnsjg"
OUTPUT: {{"is_valid": false, "reason": "I couldn't understand your question. I can help you analyze public company earnings calls, 10-K filings, and news.", "question_type": "invalid", "extracted_ticker": null, "extracted_tickers": [], "rephrased_question": "", "suggested_improvements": ["What did $AAPL say about revenue in Q4?", "Compare $MSFT and $GOOGL cloud revenue", "What's the latest news on $NVDA?"], "confidence": 0.0, "quarter_reference": null, "quarter_context": null, "quarter_count": null, "data_source": null, "needs_latest_news": false, "needs_10k": false}}

QUESTION: "hello hi"
OUTPUT: {{"is_valid": false, "reason": "Hi! I'm a financial research assistant. I can help you analyze public company earnings calls, 10-K SEC filings, and company news.", "question_type": "invalid", "extracted_ticker": null, "extracted_tickers": [], "rephrased_question": "", "suggested_improvements": ["What guidance did $TSLA provide for next quarter?", "Show me $AAPL's executive compensation from 10-K", "What are tech companies saying about AI?"], "confidence": 0.0, "quarter_reference": null, "quarter_context": null, "quarter_count": null, "data_source": null, "needs_latest_news": false, "needs_10k": false}}

QUESTION: "What's a good recipe for pasta?"
OUTPUT: {{"is_valid": false, "reason": "I can only help with public company financial data. I have access to earnings call transcripts, 10-K filings, and company news.", "question_type": "invalid", "extracted_ticker": null, "extracted_tickers": [], "rephrased_question": "", "suggested_improvements": ["What did $AAPL report in their latest earnings?", "Compare profit margins across tech companies", "What risks did $META disclose in their 10-K?"], "confidence": 0.0, "quarter_reference": null, "quarter_context": null, "quarter_count": null, "data_source": null, "needs_latest_news": false, "needs_10k": false}}

QUESTION: "How do I get a home loan?"
OUTPUT: {{"is_valid": false, "reason": "I don't have data on personal finance or loans. I specialize in public company financial analysis using earnings calls, 10-K filings, and news.", "question_type": "invalid", "extracted_ticker": null, "extracted_tickers": [], "rephrased_question": "", "suggested_improvements": ["What did banks like $JPM say about lending in earnings?", "Compare $BAC and $WFC financial performance", "What's in $GS latest 10-K filing?"], "confidence": 0.0, "quarter_reference": null, "quarter_context": null, "quarter_count": null, "data_source": null, "needs_latest_news": false, "needs_10k": false}}

**DATA SOURCE SELECTION GUIDE:**

Use semantic understanding to pick the BEST source for the question's intent:

**10-K Filings** - Choose when question is about:
- Comprehensive annual data, full-year figures, audited financials
- Balance sheets, assets, liabilities, equity, debt structure
- Executive/CEO compensation, salaries, stock awards (ONLY in 10-K)
- Risk factors, legal matters, regulatory disclosures
- Detailed segment breakdowns, business descriptions
- Multi-year historical comparisons

**Earnings Transcripts** - Choose when question is about:
- Quarterly results, recent quarter performance
- Management commentary, what executives said/discussed
- Forward guidance, next quarter/year outlook
- Analyst questions and management responses
- Strategic initiatives mentioned in earnings calls
- Quarter-over-quarter trends, sequential changes

**Latest News** - Choose when question is about:
- Very recent events (days/weeks old)
- Breaking news, current developments
- Market reactions, stock movements
- Recent announcements not yet in filings

**Hybrid** - Choose when:
- Question needs comprehensive view from multiple sources
- Comparing different time periods or data types
- User explicitly wants multiple perspectives

**KEY PRINCIPLE**: Route based on what information type would BEST answer the question, not based on specific keywords. Consider the user's underlying intent."""
                
                # Add conversation context example only when context is present
                if has_conversation_context:
                    analysis_prompt += """

CONVERSATION CONTEXT EXAMPLE (since you have conversation context above):
QUESTION: "Compare latest quarter of all companies mentioned above"
CONVERSATION CONTEXT: User previously asked about $MSFT, $GOOGL, and $INTC
OUTPUT: {{"is_valid": true, "reason": "Valid earnings question referencing previous conversation", "question_type": "multiple_companies", "extracted_ticker": "MSFT", "extracted_tickers": ["MSFT", "GOOGL", "INTC"], "rephrased_question": "key financial results and business performance highlights", "suggested_improvements": [], "confidence": 0.9, "quarter_reference": "latest", "quarter_context": "latest", "quarter_count": null}}"""
                
                analysis_prompt += """

RESPOND WITH VALID JSON ONLY. NO EXPLANATIONS OR ADDITIONAL TEXT."""

                # Add retry-specific instructions for subsequent attempts
                if attempt > 0:
                    analysis_prompt += f"""

RETRY ATTEMPT {attempt + 1}: Previous response was invalid JSON.
CRITICAL: Return ONLY valid JSON matching the exact structure above.
Double-check: no trailing commas, proper quotes, all 11 fields present.
Example: {{"is_valid": true, "reason": "Valid question", "question_type": "specific_company", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "rephrased_question": "concept-focused question without company or time", "suggested_improvements": ["Improvement"], "confidence": 0.9, "quarter_reference": null, "quarter_context": "latest", "quarter_count": null}}"""

                cerebras_model = self.config.get("cerebras_model", "qwen-3-235b-a22b-instruct-2507")
                rag_logger.info(f"🤖 Sending question to Cerebras model: {cerebras_model} (attempt {attempt + 1}/{max_retries})")
                
                # Build context-aware system message
                if has_conversation_context:
                    rag_logger.info(f"🧠 Using CONVERSATION CONTEXT MODE - will emphasize ticker extraction from conversation history")
                    system_message = "You are a JSON-only response assistant for financial data routing. Respond with valid JSON only. Extract tickers from $TICKER format. CRITICAL: Conversation context is provided - extract ALL relevant tickers from the conversation history when the question references previous companies. For multiple companies, rephrased_question must be GENERIC without specific company names. YEAR ONLY questions = quarter_context: 'multiple', quarter_count: 4. SEMANTIC ROUTING: Choose data_source based on what information type BEST answers the question's intent: '10k' for annual data, financial statements, compensation, risk factors, audited figures; 'earnings_transcripts' for quarterly results, management commentary, guidance, analyst Q&A; 'latest_news' for recent events, breaking news, current developments; 'hybrid' when multiple perspectives needed. Executive compensation is ONLY in 10-K filings. No explanations or additional text."
                else:
                    rag_logger.info(f"🎯 Using STANDARD MODE - no conversation context available")
                    system_message = "You are a JSON-only response assistant for financial data routing. Respond with valid JSON only. Extract tickers from $TICKER format. For multiple companies, rephrased_question must be GENERIC without specific company names. YEAR ONLY questions = quarter_context: 'multiple', quarter_count: 4. SEMANTIC ROUTING: Choose data_source based on what information type BEST answers the question's intent: '10k' for annual data, financial statements, compensation, risk factors, audited figures; 'earnings_transcripts' for quarterly results, management commentary, guidance, analyst Q&A; 'latest_news' for recent events, breaking news, current developments; 'hybrid' when multiple perspectives needed. Executive compensation is ONLY in 10-K filings. No explanations or additional text."
                
                start_time = time.time()
                cerebras_model = self.config.get("cerebras_model", "qwen-3-235b-a22b-instruct-2507")

                # Log to Logfire with span for observability (full prompts)
                if LOGFIRE_AVAILABLE and logfire:
                    with logfire.span(
                        "cerebras.question_analysis",
                        model=cerebras_model,
                        question=question,
                        system_prompt=system_message,
                        user_prompt=analysis_prompt,
                        has_conversation_context=has_conversation_context,
                        attempt=attempt + 1,
                        max_retries=max_retries
                    ):
                        response = self.client.chat.completions.create(
                            model=cerebras_model,
                            messages=[
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": analysis_prompt}
                            ],
                            max_completion_tokens=1000,
                            temperature=0.1
                        )
                        call_time = time.time() - start_time

                        # Log completion details with full response
                        logfire.info(
                            "cerebras.question_analysis.completion",
                            model=cerebras_model,
                            duration_seconds=call_time,
                            response=response.choices[0].message.content if response.choices else None,
                            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
                            completion_tokens=response.usage.completion_tokens if response.usage else None,
                            total_tokens=response.usage.total_tokens if response.usage else None
                        )
                else:
                    response = self.client.chat.completions.create(
                        model=cerebras_model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": analysis_prompt}
                        ],
                        max_completion_tokens=1000,
                        temperature=0.1
                    )
                    call_time = time.time() - start_time

                rag_logger.info(f"✅ Received response from Cerebras (tokens: {response.usage.total_tokens if response.usage else 'unknown'})")
                
                # Parse JSON response
                analysis_text = response.choices[0].message.content.strip()
                rag_logger.info(f"📝 Raw Cerebras response length: {len(analysis_text)} characters")
                rag_logger.info(f"📝 Raw Cerebras response (first 500 chars): {analysis_text[:500]}")
                if len(analysis_text) == 0:
                    rag_logger.error("❌ CRITICAL: Model returned empty response!")
                    raise Exception("Model returned empty response - this indicates a model or prompt issue")
                
                # Clean up the response (remove any markdown formatting)
                if analysis_text.startswith("```json"):
                    analysis_text = analysis_text[7:]
                    rag_logger.info("🧹 Removed ```json prefix from response")
                if analysis_text.endswith("```"):
                    analysis_text = analysis_text[:-3]
                    rag_logger.info("🧹 Removed ``` suffix from response")
                
                # Try to parse JSON with repair attempts
                try:
                    analysis_result = parse_json_with_repair(analysis_text, attempt, QuestionAnalysisResult, rag_logger)
                    rag_logger.info(f"✅ Successfully parsed JSON analysis result")
                    rag_logger.info(f"📊 Analysis result: valid={analysis_result.get('is_valid')}, ticker={analysis_result.get('extracted_ticker')}, type={analysis_result.get('question_type')}, needs_latest_news={analysis_result.get('needs_latest_news', False)}")

                    # DEBUG: Log data_source from Cerebras
                    rag_logger.info(f"🔍 DEBUG [CEREBRAS RESPONSE]: data_source={analysis_result.get('data_source')}, needs_10k={analysis_result.get('needs_10k')}")

                    # Add original question
                    analysis_result["original_question"] = question
                except Exception as parse_error:
                    # If validation fails but we have the raw JSON, try to extract needs_latest_news
                    rag_logger.warning(f"⚠️ Validation failed but attempting to extract needs_latest_news from raw response")
                    import json as json_lib
                    try:
                        raw_json = json_lib.loads(analysis_text)
                        needs_news = raw_json.get('needs_latest_news', False)
                        rag_logger.info(f"📰 Extracted needs_latest_news={needs_news} from raw JSON")
                        # Re-raise to continue with normal error handling
                        raise parse_error
                    except:
                        raise parse_error
                
                # Log quarter detection results for debugging
                rag_logger.info(f"🔍 Quarter detection results: context={analysis_result.get('quarter_context')}, count={analysis_result.get('quarter_count')}, reference={analysis_result.get('quarter_reference')}")
                
                # Log ticker extraction results for debugging (especially important for conversation context)
                extracted_tickers = analysis_result.get('extracted_tickers', [])
                rephrased_question = analysis_result.get('rephrased_question', '')
                question_type = analysis_result.get('question_type', '')
                
                if extracted_tickers:
                    rag_logger.info(f"🎯 Extracted tickers: {extracted_tickers}")
                    if conversation_context:
                        rag_logger.info(f"🎯 Tickers were extracted with conversation context available")
                    
                    # Verify generic rephrasing for multiple companies
                    if len(extracted_tickers) > 1 and question_type == 'multiple_companies':
                        # Check if rephrased question contains company names (it shouldn't)
                        contains_company_names = any(ticker.lower() in rephrased_question.lower() for ticker in extracted_tickers)
                        if contains_company_names:
                            rag_logger.warning(f"⚠️ Multiple companies detected but rephrased question contains company names: '{rephrased_question}'")
                        else:
                            rag_logger.info(f"✅ Multiple companies with generic rephrased question: '{rephrased_question}'")
                else:
                    rag_logger.warning(f"⚠️ No tickers extracted from question: '{question}'")
                    if conversation_context:
                        rag_logger.warning(f"⚠️ Conversation context was available but no tickers extracted!")

                # Log question analysis to Logfire with original and rephrased questions
                if LOGFIRE_AVAILABLE and logfire:
                    logfire.info(
                        "question.analysis.complete",
                        original_question=question,
                        rephrased_question=analysis_result.get('rephrased_question', ''),
                        is_valid=analysis_result.get('is_valid', False),
                        question_type=analysis_result.get('question_type', ''),
                        data_source=analysis_result.get('data_source', 'earnings_transcripts'),
                        tickers=extracted_tickers,
                        needs_10k=analysis_result.get('needs_10k', False),
                        needs_latest_news=analysis_result.get('needs_latest_news', False),
                        quarter_context=analysis_result.get('quarter_context', 'latest'),
                        confidence=analysis_result.get('confidence', 0)
                    )

                return analysis_result
                
            except json.JSONDecodeError as e:
                rag_logger.error(f"❌ JSON parsing failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay  # Fixed delay
                    rag_logger.info(f"🔄 Retrying in {delay} seconds... (attempt {attempt + 2}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    rag_logger.error(f"💥 All {max_retries} attempts failed. Last error: {e}")
                    rag_logger.error(f"📝 Last response text: {analysis_text[:500]}...")
                    
                    # Try to extract fields from the raw response - PRESERVE is_valid from LLM!
                    rag_logger.warning("🔄 Attempting to extract response from raw JSON despite validation errors")

                    try:
                        import json as json_lib
                        raw_json = json_lib.loads(analysis_text)

                        # CRITICAL: Preserve is_valid from LLM response - don't override!
                        fallback_is_valid = raw_json.get('is_valid', True)
                        fallback_reason = raw_json.get('reason', 'Question analysis completed')
                        fallback_question_type = raw_json.get('question_type', 'multiple_companies')
                        fallback_needs_news = raw_json.get('needs_latest_news', False)
                        fallback_needs_10k = raw_json.get('needs_10k', False)
                        fallback_data_source = raw_json.get('data_source')
                        fallback_rephrased = raw_json.get('rephrased_question', question)
                        fallback_suggestions = raw_json.get('suggested_improvements', [])
                        fallback_confidence = raw_json.get('confidence', 0.5)
                        fallback_tickers = raw_json.get('extracted_tickers', [])
                        fallback_ticker = raw_json.get('extracted_ticker')

                        rag_logger.info(f"📋 Extracted from raw response: is_valid={fallback_is_valid}, question_type={fallback_question_type}, data_source={fallback_data_source}")

                        fallback_response = {
                            "is_valid": fallback_is_valid,
                            "reason": fallback_reason,
                            "question_type": fallback_question_type,
                            "extracted_ticker": fallback_ticker,
                            "extracted_tickers": fallback_tickers,
                            "rephrased_question": fallback_rephrased,
                            "suggested_improvements": fallback_suggestions,
                            "confidence": fallback_confidence,
                            "quarter_reference": raw_json.get('quarter_reference'),
                            "quarter_context": raw_json.get('quarter_context'),
                            "quarter_count": raw_json.get('quarter_count'),
                            "data_source": fallback_data_source,
                            "needs_latest_news": fallback_needs_news,
                            "needs_10k": fallback_needs_10k,
                            "original_question": question
                        }

                        rag_logger.info(f"✅ Fallback response created from raw JSON: is_valid={fallback_is_valid}, question_type={fallback_question_type}")
                        return fallback_response

                    except Exception as parse_err:
                        rag_logger.error(f"❌ Could not parse raw JSON: {parse_err}")
                        # Only if we truly can't parse anything, return a generic error
                        return {
                            "is_valid": False,
                            "reason": "Unable to analyze your question. Please try rephrasing it.",
                            "question_type": "invalid",
                            "extracted_ticker": None,
                            "extracted_tickers": [],
                            "rephrased_question": "",
                            "suggested_improvements": ["Try asking about a specific company like $AAPL or $MSFT"],
                            "confidence": 0.0,
                            "quarter_reference": None,
                            "quarter_context": None,
                            "quarter_count": None,
                            "data_source": None,
                            "needs_latest_news": False,
                            "needs_10k": False,
                            "original_question": question
                        }
            except Exception as e:
                rag_logger.error(f"❌ Question analysis failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay  # Fixed delay
                    rag_logger.info(f"🔄 Retrying in {delay} seconds... (attempt {attempt + 2}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    rag_logger.error(f"💥 All {max_retries} attempts failed. Last error: {e}")
                    
                    # Try to extract fields from the raw response - PRESERVE is_valid from LLM!
                    rag_logger.warning("🔄 Attempting to extract response from raw JSON despite general errors")

                    try:
                        import json as json_lib
                        raw_json = json_lib.loads(analysis_text)

                        # CRITICAL: Preserve is_valid from LLM response - don't override!
                        fallback_is_valid = raw_json.get('is_valid', True)
                        fallback_reason = raw_json.get('reason', 'Question analysis completed')
                        fallback_question_type = raw_json.get('question_type', 'multiple_companies')
                        fallback_needs_news = raw_json.get('needs_latest_news', False)
                        fallback_needs_10k = raw_json.get('needs_10k', False)
                        fallback_data_source = raw_json.get('data_source')
                        fallback_rephrased = raw_json.get('rephrased_question', question)
                        fallback_suggestions = raw_json.get('suggested_improvements', [])
                        fallback_confidence = raw_json.get('confidence', 0.5)
                        fallback_tickers = raw_json.get('extracted_tickers', [])
                        fallback_ticker = raw_json.get('extracted_ticker')

                        rag_logger.info(f"📋 Extracted from raw response: is_valid={fallback_is_valid}, question_type={fallback_question_type}, data_source={fallback_data_source}")

                        fallback_response = {
                            "is_valid": fallback_is_valid,
                            "reason": fallback_reason,
                            "question_type": fallback_question_type,
                            "extracted_ticker": fallback_ticker,
                            "extracted_tickers": fallback_tickers,
                            "rephrased_question": fallback_rephrased,
                            "suggested_improvements": fallback_suggestions,
                            "confidence": fallback_confidence,
                            "quarter_reference": raw_json.get('quarter_reference'),
                            "quarter_context": raw_json.get('quarter_context'),
                            "quarter_count": raw_json.get('quarter_count'),
                            "data_source": fallback_data_source,
                            "needs_latest_news": fallback_needs_news,
                            "needs_10k": fallback_needs_10k,
                            "original_question": question
                        }

                        rag_logger.info(f"✅ Fallback response created from raw JSON: is_valid={fallback_is_valid}, question_type={fallback_question_type}")
                        return fallback_response

                    except Exception as parse_err:
                        rag_logger.error(f"❌ Could not parse raw JSON: {parse_err}")
                        # Only if we truly can't parse anything, return a generic error
                        return {
                            "is_valid": False,
                            "reason": "Unable to analyze your question. Please try rephrasing it.",
                            "question_type": "invalid",
                            "extracted_ticker": None,
                            "extracted_tickers": [],
                            "rephrased_question": "",
                            "suggested_improvements": ["Try asking about a specific company like $AAPL or $MSFT"],
                            "confidence": 0.0,
                            "quarter_reference": None,
                            "quarter_context": None,
                            "quarter_count": None,
                            "data_source": None,
                            "needs_latest_news": False,
                            "needs_10k": False,
                            "original_question": question
                        }
        
        # This should never be reached, but just in case
        raise Exception("Unexpected error in analyze_question retry loop")

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 3: QUARTER DETERMINATION & RESOLUTION
    # ═════════════════════════════════════════════════════════════════════

    def determine_target_quarter(self, analysis: Dict[str, Any], ticker: str = None) -> str:
        """Determine the target quarter based on question analysis."""
        quarter_reference = analysis.get('quarter_reference')
        quarter_context = analysis.get('quarter_context', 'latest')
        quarter_count = analysis.get('quarter_count')
        
        # Resolve 'latest' quarter references first
        if quarter_reference == 'latest':
            # Use database_manager to resolve latest quarter
            if self.database_manager:
                resolved_quarter = self.database_manager.resolve_latest_quarter_reference(quarter_reference, ticker)
                if resolved_quarter != "NO_QUARTERS_AVAILABLE":
                    return resolved_quarter
            # Fallback to configured latest quarter
            latest_quarter = self.config.get_latest_quarter()
            if latest_quarter:
                return latest_quarter
            else:
                return "NO_QUARTERS_AVAILABLE"
        
        # If specific quarter is mentioned, try to match it
        if quarter_reference:
            available_quarters = self.config.get('available_quarters', [])
            
            # Direct match (should work now since LLM uses database format)
            if quarter_reference in available_quarters:
                return quarter_reference
            
            # Quarter requested but not available - return special marker for clear error
            return f"UNAVAILABLE_QUARTER:{quarter_reference}"
        
        # Handle multiple quarters (e.g., "last 3 quarters" or year-only like "2024")
        if quarter_context == 'multiple' and quarter_count:
            # Special case: quarter_count=4 usually means whole year
            if quarter_count == 4:
                return 'year_all'  # Special marker for full year queries
            return 'multiple'  # General marker for multiple quarter queries
        
        # Handle context-based quarter selection
        if quarter_context == 'previous':
            available_quarters = self.config.get('available_quarters', [])
            if len(available_quarters) >= 2:
                # If we have multiple quarters, previous would be the second-to-last
                return available_quarters[-2]
            elif len(available_quarters) == 1:
                return self.config.get_latest_quarter()
            else:
                return "NO_QUARTERS_AVAILABLE"
        
        # Handle latest quarter (explicit case)
        if quarter_context == 'latest':
            # Check if quarter_reference is specifically "latest"
            if quarter_reference == 'latest':
                available_quarters = self.config.get('available_quarters', [])
                if available_quarters:
                    return self.config.get_latest_quarter()  # Latest quarter is first in the list
                else:
                    return "NO_QUARTERS_AVAILABLE"
            else:
                available_quarters = self.config.get('available_quarters', [])
                if available_quarters:
                    return self.config.get_latest_quarter()  # Latest quarter is first in the list
                else:
                    return "NO_QUARTERS_AVAILABLE"
        
        # Default to latest quarter if available
        available_quarters = self.config.get('available_quarters', [])
        if available_quarters:
            return available_quarters[0]
        else:
            return "NO_QUARTERS_AVAILABLE"

    def add_to_conversation_memory(self, conversation_id: str, message: str, role: str = "user"):
        """Add a message to conversation memory."""
        self.conversation_memory.add_message(conversation_id, message, role)
    
    def get_quarters_to_search(self, analysis: Dict[str, Any]) -> List[str]:
        """Get list of quarters to search based on question analysis."""
        quarter_context = analysis.get('quarter_context', 'latest')
        quarter_count = analysis.get('quarter_count')
        available_quarters = self.config.get('available_quarters', [])
        
        # Debug logging
        rag_logger.info(f"🔍 get_quarters_to_search debug (instance: {getattr(self, 'instance_id', 'unknown')}):")
        rag_logger.info(f"   quarter_context: {quarter_context}")
        rag_logger.info(f"   quarter_count: {quarter_count}")
        rag_logger.info(f"   available_quarters: {available_quarters}")
        rag_logger.info(f"   available_quarters length: {len(available_quarters)}")
        
        # Handle multiple quarters (e.g., "last 3 quarters", "2024", "last 1 year")
        if quarter_context == 'multiple' and quarter_count:
            # Special case: quarter_count=4 could mean:
            # 1. "last 1 year" (last 4 quarters) - should use company-specific quarters
            # 2. Year-only query like "2024" or "2025" - all quarters in that specific year
            if quarter_count == 4:
                quarter_reference = analysis.get('quarter_reference')
                # Check if this is a year-only query (has explicit year reference like "2024_all" or "2025")
                # A specific quarter like "2024_q4" is NOT a year-only query - it's a specific quarter reference
                is_year_only = quarter_reference and (
                    '_all' in quarter_reference or  # e.g., "2024_all"
                    (quarter_reference.isdigit() and len(quarter_reference) == 4)  # e.g., "2024" or "2025"
                ) and '_q' not in quarter_reference  # Exclude specific quarters like "2024_q4"
                
                if is_year_only:
                    # Extract year from quarter_reference
                    if '_all' in quarter_reference:
                        year = quarter_reference.split('_')[0]
                    else:
                        year = quarter_reference
                    rag_logger.info(f"  🗓️ YEAR-ONLY DETECTED: Looking for all quarters in year {year}")
                    # Find all quarters for this specific year
                    year_quarters = [q for q in available_quarters if q.startswith(year + '_')]
                    year_quarters.sort(reverse=True)  # Sort newest first
                    rag_logger.info(f"   year-only quarters result: {year_quarters}")
                    return year_quarters
                else:
                    # This is "last 1 year" (last 4 quarters) - use company-specific quarters if available
                    ticker = analysis.get('extracted_ticker')
                    if ticker and self.database_manager:
                        # Get company-specific last 4 quarters from database
                        company_quarters = self.database_manager.get_last_n_quarters_for_company(ticker, 4)
                        if company_quarters:
                            rag_logger.info(f"   ✅ Company-specific last 4 quarters (1 year) for {ticker}: {company_quarters}")
                            return company_quarters
                        else:
                            rag_logger.warning(f"   ⚠️ No company-specific quarters found for {ticker}, falling back to general quarters")
                    
                    # Fallback to general available quarters (when no ticker or company-specific query failed)
                    result = self._get_last_n_quarters_business_logic(available_quarters, 4)
                    rag_logger.info(f"   general last 4 quarters result: {result}")
                    return result
            else:
                # For multiple quarters (not 4), always try to get company-specific quarters if ticker is available
                ticker = analysis.get('extracted_ticker')
                if ticker and self.database_manager:
                    # Get company-specific last N quarters from database
                    # This finds the latest quarter for this specific company and gets N quarters going back
                    company_quarters = self.database_manager.get_last_n_quarters_for_company(ticker, quarter_count)
                    if company_quarters:
                        rag_logger.info(f"   ✅ Company-specific last {quarter_count} quarters for {ticker}: {company_quarters}")
                        return company_quarters
                    else:
                        rag_logger.warning(f"   ⚠️ No company-specific quarters found for {ticker}, falling back to general quarters")
                
                # Fallback to general available quarters (when no ticker or company-specific query failed)
                # Use the actual latest quarter from available_quarters (already sorted DESC)
                result = self._get_last_n_quarters_business_logic(available_quarters, quarter_count)
                rag_logger.info(f"   general multiple quarters result: {result}")
                return result
        
        # For single quarter queries, return the determined target quarter
        # Extract ticker from analysis for company-specific latest quarter resolution
        ticker = analysis.get('extracted_ticker')
        target_quarter = self.determine_target_quarter(analysis, ticker)
        
        # Handle special error cases
        if target_quarter == 'multiple':
            # Fallback to all available quarters
            return available_quarters
        elif target_quarter.startswith('UNAVAILABLE_QUARTER:'):
            # Return empty list to trigger clear error message
            return []
        elif target_quarter == 'NO_QUARTERS_AVAILABLE':
            # Return empty list to trigger clear error message
            return []
        else:
            return [target_quarter]
    
    def _get_last_n_quarters_business_logic(self, available_quarters: List[str], n: int) -> List[str]:
        """
        Get the last N quarters from available quarters.
        
        This method assumes available_quarters is already sorted in reverse chronological order
        (year DESC, quarter DESC), so the first quarter is the latest. It simply returns
        the first N quarters from the list.
        
        Args:
            available_quarters: List of available quarters (sorted DESC, latest first)
            n: Number of quarters to return
            
        Returns:
            List of the last N quarters (latest first)
        """
        if not available_quarters:
            return []
        
        # available_quarters is already sorted in reverse chronological order (latest first)
        # So we just take the first N quarters
        result = available_quarters[:n] if len(available_quarters) >= n else available_quarters
        rag_logger.info(f"🔄 Last {n} quarters from available quarters: {result}")
        return result

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 4: QUESTION VALIDATION & PROCESSING
    # ═════════════════════════════════════════════════════════════════════

    async def validate_question(self, question: str, conversation_id: str = None) -> Dict[str, Any]:
        """Validate a question and return a user-friendly response."""
        analysis = await self.analyze_question(question, conversation_id)
        
        # Check if question is invalid based on AI's assessment
        is_valid = analysis.get("is_valid", False)
        
        # If the AI marked it as invalid, reject it
        if not is_valid:
            # Use the reason directly from analysis - it already explains what we can help with
            message = analysis.get('reason', 'I can help you analyze public company earnings calls, 10-K SEC filings, and company news.')
            suggestions = analysis.get('suggested_improvements', [
                "What did $AAPL say about revenue in Q4?",
                "Compare $MSFT and $GOOGL cloud revenue",
                "What's the latest news on $NVDA?"
            ])

            return {
                "status": "rejected",
                "message": message,
                "suggestions": suggestions,
                "question_type": analysis["question_type"],
                "original_question": question
            }
        
        # Prepare response
        response = {
            "status": "accepted",
            "rephrased_question": analysis["rephrased_question"],
            "question_type": analysis["question_type"],
            "confidence": analysis.get("confidence", 0.8),  # Default confidence if not provided
            "original_question": question,
            # Include quarter-related fields
            "quarter_context": analysis.get("quarter_context"),
            "quarter_count": analysis.get("quarter_count"),
            "quarter_reference": analysis.get("quarter_reference"),
            # Include data source routing (PRIMARY FIELD)
            "data_source": analysis.get("data_source", "earnings_transcripts"),
            # Include news-related fields (legacy, use data_source instead)
            "needs_latest_news": analysis.get("needs_latest_news", False),
            # Include 10-K fields (legacy, use data_source instead)
            "needs_10k": analysis.get("needs_10k", False)
        }
        
        if analysis.get("extracted_tickers"):
            response["extracted_tickers"] = analysis["extracted_tickers"]
            response["extracted_ticker"] = analysis["extracted_tickers"][0]
        else:
            response["extracted_ticker"] = analysis.get("extracted_ticker")
            response["extracted_tickers"] = [analysis.get("extracted_ticker")] if analysis.get("extracted_ticker") else []

        # DEBUG: Log data_source in validation response
        rag_logger.info(f"🔍 DEBUG [VALIDATION RESPONSE]: data_source={response.get('data_source')}, needs_10k={response.get('needs_10k')}")

        return response
    
    async def process_question(self, question: str, conversation_id: str = None) -> Dict[str, Any]:
        """Complete question processing pipeline."""
        # First validate the question
        validation = await self.validate_question(question, conversation_id)
        
        if validation["status"] == "rejected":
            return validation

        # If accepted, return the processed question
        result = {
            "status": "processed",
            "original_question": question,
            "processed_question": validation["rephrased_question"],
            "question_type": validation["question_type"],
            "extracted_ticker": validation["extracted_ticker"],
            "extracted_tickers": validation.get("extracted_tickers", [validation["extracted_ticker"]] if validation["extracted_ticker"] else []),
            "confidence": validation["confidence"],
            # Include quarter-related fields
            "quarter_context": validation.get("quarter_context"),
            "quarter_count": validation.get("quarter_count"),
            "quarter_reference": validation.get("quarter_reference"),
            # Include data source routing (PRIMARY FIELD)
            "data_source": validation.get("data_source", "earnings_transcripts"),
            # Include news-related fields (legacy, use data_source instead)
            "needs_latest_news": validation.get("needs_latest_news", False),
            # Include 10-K fields (legacy, use data_source instead)
            "needs_10k": validation.get("needs_10k", False)
        }

        # DEBUG: Log final data_source being returned
        rag_logger.info(f"🔍 DEBUG [PROCESS_QUESTION FINAL]: data_source={result.get('data_source')}, needs_10k={result.get('needs_10k')}")

        return result

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 5: TICKER-SPECIFIC QUESTION CREATION
    # ═════════════════════════════════════════════════════════════════════

    def create_ticker_specific_question(self, original_question: str, ticker: str) -> str:
        """Create a ticker-specific rephrased question for better search results."""
        rag_logger.info(f"🎯 Creating ticker-specific question for {ticker}")

        # Get ticker-specific prompt from centralized prompts
        ticker_prompt = get_ticker_rephrasing_prompt(original_question, ticker)

        try:
            # Detailed LLM stage logging for ticker-specific rephrasing
            rag_logger.info(f"🤖 ===== TICKER REPHRASING LLM CALL =====")
            cerebras_model = self.config.get("cerebras_model", "qwen-3-235b-a22b-instruct-2507")
            rag_logger.info(f"🔍 Model: {cerebras_model}")
            rag_logger.info(f"📊 Max tokens: 200")
            rag_logger.info(f"🌡️ Temperature: 0.3")
            rag_logger.info(f"🎯 Ticker: {ticker}")
            rag_logger.info(f"📝 Original question: {original_question}")
            rag_logger.info(f"📋 Ticker prompt length: {len(ticker_prompt)} characters")
            rag_logger.info(f"📋 Ticker prompt preview: {ticker_prompt[:200]}...")

            start_time = time.time()
            response = self.client.chat.completions.create(
                model=cerebras_model,
                messages=[
                    {"role": "system", "content": TICKER_REPHRASING_SYSTEM_PROMPT},
                    {"role": "user", "content": ticker_prompt}
                ],
                max_completion_tokens=200,
                temperature=0.3
            )
            call_time = time.time() - start_time
            
            ticker_question = response.choices[0].message.content.strip()
            
            # Detailed response logging
            rag_logger.info(f"✅ ===== TICKER REPHRASING LLM RESPONSE ===== (call time: {call_time:.3f}s)")
            rag_logger.info(f"📊 Response tokens used: {response.usage.total_tokens if response.usage else 'unknown'}")
            rag_logger.info(f"📊 Prompt tokens: {response.usage.prompt_tokens if response.usage else 'unknown'}")
            rag_logger.info(f"📊 Completion tokens: {response.usage.completion_tokens if response.usage else 'unknown'}")
            if hasattr(response, 'finish_reason'):
                rag_logger.info(f"🏁 Finish reason: {response.finish_reason}")
            rag_logger.info(f"📝 Rephrased question: {ticker_question}")
            rag_logger.info(f"✅ Created ticker-specific question: '{ticker_question}'")
            return ticker_question
            
        except Exception as e:
            rag_logger.error(f"❌ Error creating ticker-specific question: {e}")
            raise Exception(f"Failed to create ticker-specific question: {e}")
