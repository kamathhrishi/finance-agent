#!/usr/bin/env python3
"""
Centralized Prompts for the Agent System

This file contains all LLM prompts used across the agent system for:
- Ticker-specific rephrasing (multi-ticker queries)
- Quarter synthesis (multi-quarter responses)
- Context-aware follow-ups (iterative improvement)
"""

# ============================================================================
# TICKER-SPECIFIC REPHRASING PROMPTS
# ============================================================================

TICKER_REPHRASING_SYSTEM_PROMPT = """You are a financial analyst assistant that creates ticker-specific search queries for earnings transcripts. Be precise and focused."""

def get_ticker_rephrasing_prompt(original_question: str, ticker: str) -> str:
    """
    Generate prompt for rephrasing question to be ticker-specific.

    Args:
        original_question: Original user question
        ticker: Ticker symbol to focus on

    Returns:
        Formatted ticker rephrasing prompt
    """
    return f"""Rephrase this question to be specific to {ticker} while maintaining the core business topic. Make it more targeted for searching {ticker}'s earnings transcripts.

Original Question: "{original_question}"
Target Ticker: {ticker}

Instructions:
1. Keep the core business topic (revenue, AI, guidance, etc.)
2. Make it specific to {ticker}
3. Remove other ticker references but keep {ticker}
4. Make it a clear search query for {ticker}'s earnings data
5. Keep it concise and focused

Examples:
- "How do $AAPL and $MSFT compare on AI?" → "What did {ticker} say about AI investments?"
- "What were the revenue highlights for $AAPL and $MSFT?" → "What were {ticker}'s revenue highlights?"
- "How do companies compare on guidance?" → "What guidance did {ticker} provide?"

Respond with ONLY the rephrased question, no other text."""


# ============================================================================
# PARALLEL QUARTER SYNTHESIS PROMPTS
# ============================================================================

QUARTER_SYNTHESIS_SYSTEM_PROMPT = """You are a financial analyst assistant that synthesizes multi-quarter earnings data into comprehensive, well-organized responses. Your synthesis must DIRECTLY ANSWER the original question. ALWAYS include ALL financial figures from ALL quarters with EXACT numbers, percentages, and dollar amounts. ALWAYS maintain quarter and company metadata in human-friendly format (Q1 2025, Q2 2025). ALWAYS cite guidance, projections, and forward-looking statements. Show trends, comparisons, and complete financial progression across periods with specific quarter-over-quarter metrics. Be EXTREMELY DETAILED and ELABORATE with all quantitative data. NEVER omit any financial figures - if a number appeared in any quarter response, include it in the synthesis. Preserve all nuances, executive quotes, strategic insights, and contextual information. Your response should be analyst-quality, comprehensive, and directly address what was asked."""

def get_quarter_synthesis_prompt(question: str, quarter_responses: list, company_name: str,
                                quarters_human: list) -> str:
    """
    Generate prompt for synthesizing multiple quarter responses into one answer.

    Args:
        question: Original user question
        quarter_responses: List of quarter response dictionaries
        company_name: Company name or ticker
        quarters_human: List of human-friendly quarter labels (e.g., ['Q1 2025', 'Q2 2025'])

    Returns:
        Formatted quarter synthesis prompt
    """
    # Build the context with labeled quarter responses
    context_parts = []
    for qr in quarter_responses:
        quarter_label = f"Q{qr['quarter']} {qr['year']}"
        context_parts.append(f"### {quarter_label} Response:\n{qr['answer']}")

    context = "\n\n".join(context_parts)

    return f"""You are a financial analyst assistant. You have detailed responses for multiple quarters regarding the same question. Your task is to synthesize these into ONE comprehensive, well-organized answer that DIRECTLY ANSWERS THE ORIGINAL QUESTION.

Company: {company_name}
Quarters Analyzed: {', '.join(quarters_human)} ({len(quarter_responses)} quarters total)

Original Question: {question}

Individual Quarter Responses (each contains detailed analysis from that specific quarter):
{context}

Instructions for Synthesis - READ CAREFULLY:

**PRIMARY GOAL**: Directly answer the original question using integrated data from ALL quarters

1. **Answer the Question First** - Start with a direct answer to what was asked, then provide supporting details

2. **Create a UNIFIED, COMPREHENSIVE response** that integrates information from ALL quarters
   - Don't just concatenate responses - synthesize them into a cohesive narrative
   - Show the complete picture across the time period
   - Make it read as ONE analysis, not separate quarter reports

3. **ALWAYS Maintain Quarter & Company Metadata** - CRITICAL for context:
   - Reference specific quarters when citing data (e.g., "In Q1 2025, {company_name} reported...")
   - Use human-friendly format: "Q1 2025", "Q2 2025", "Q4 2024" (NOT "2025_q1")
   - Always mention {company_name} by name when discussing metrics
   - Provide source attribution: "According to {company_name}'s Q1 2025 earnings transcript..."
   - Track and display quarter-over-quarter changes with specific quarter references

4. **CRITICAL - ALWAYS MENTION ALL FINANCIAL FIGURES & PROJECTIONS**:
   - **Include EVERY financial number from EVERY quarter** - NEVER omit any figure
   - Provide EXACT dollar amounts, percentages, and units - never round or approximate
   - Include ALL metrics: revenue, profit, margins, growth rates, EPS, cash flow, EBITDA, etc.
   - **ALWAYS cite guidance, projections, and forward-looking statements** from any quarter
   - Show quarter-over-quarter progression with SPECIFIC numbers (e.g., "Revenue: Q1 $5.2B → Q2 $5.8B (+11.5%)")
   - Include ALL comparative metrics (YoY, QoQ, sequential changes)
   - Detail cost structures, expense breakdowns, margin analyses
   - Break down segment-level, product-level, and geographic financials from each quarter
   - **DEFAULT BEHAVIOR**: If a financial figure appears in ANY quarter response, it MUST appear in the synthesis
   - If guidance changed between quarters, show the progression with exact ranges

5. **Show Trends & Patterns Across Quarters**:
   - Highlight improvements, declines, or consistency across the period
   - Calculate and show growth trajectories with specific percentages
   - Compare performance metrics across quarters
   - Identify inflection points or significant changes
   - Show cumulative effects when relevant

6. **Structure & Organization**:
   - Choose chronological OR thematic structure based on what best answers the question
   - Use **markdown formatting** with **bold** for key metrics, bullet points for lists
   - Use clear section headers when helpful
   - Be ELABORATE and DETAILED - include all nuances and context
   - Organize logically but comprehensively

7. **Maintain All Nuances**:
   - Include executive quotes if present in quarter responses
   - Preserve strategic insights and qualitative commentary
   - Keep contextual information about market conditions, challenges, opportunities
   - Maintain any specific guidance, targets, or outlook statements
   - Include operational details, KPIs, and business metrics

8. **Avoid Repetition** - Synthesize intelligently:
   - If multiple quarters discuss the same initiative, show its evolution
   - If a metric is consistent, state it once with confirmation across quarters
   - Focus on changes, trends, and progression rather than redundant statements

9. **Be Comprehensive Yet Readable**:
   - Include ALL relevant information from ALL quarters
   - NEVER say "based on available data" or similar - you have complete quarter data
   - Leave NO financial metric unexplained or unmentioned
   - Provide COMPLETE CONTEXT for every financial figure (what it represents, why it matters, how it changed)

10. **Quality Standards**:
    - Be as ELABORATE and DETAILED as possible
    - Provide a professional, analyst-quality response
    - Use specific numbers, not generalizations
    - Support every statement with data from specific quarters

IMPORTANT REMINDER: The original question was "{question}" - make sure your synthesis directly answers this question using the multi-quarter data. Start with a clear answer, then provide comprehensive supporting analysis."""


# ============================================================================
# CONTEXT-AWARE FOLLOW-UP PROMPTS
# ============================================================================

CONTEXT_AWARE_FOLLOWUP_SYSTEM_PROMPT = """You are a financial analyst assistant. Always respond with valid JSON arrays only. No additional text or formatting."""

def get_context_aware_followup_prompt(original_question: str, current_answer: str,
                                     available_chunks: list) -> str:
    """
    Generate prompt for context-aware follow-up question generation.

    Args:
        original_question: Original user question
        current_answer: Current answer generated
        available_chunks: Available context chunks with metadata

    Returns:
        Formatted context-aware follow-up prompt
    """
    # Build context analysis
    context_analysis = ""
    if available_chunks:
        context_analysis = "\n\n"
        for i, chunk in enumerate(available_chunks[:5], 1):
            context_analysis += f"\nChunk {i}:\n"
            context_analysis += f"Text preview: {chunk.get('chunk_text', '')[:150]}...\n"
            if chunk.get('year') and chunk.get('quarter'):
                context_analysis += f"Quarter: {chunk['year']}_q{chunk['quarter']}\n"
            if chunk.get('ticker'):
                context_analysis += f"Ticker: {chunk['ticker']}\n"
            if chunk.get('distance'):
                context_analysis += f"Relevance: {chunk['distance']:.3f}\n"

    return f"""You are analyzing available context to suggest better follow-up questions for a financial research query.

Original Question: {original_question}
Current Answer: {current_answer}

Available Context:{context_analysis}

Based on the available context, suggest 2-3 specific follow-up questions that would help:
1. Find more relevant information that might be missing
2. Clarify specific aspects of the original question
3. Explore related topics that could provide better insights

**CRITICAL: Preserve temporal and contextual scope from the original question**
- If the original question mentions specific time periods (e.g., "last three quarters", "Q1 2024"), ALL follow-up questions MUST include the same time period
- If the original mentions specific companies/tickers, include them in follow-up questions
- Follow-up questions should refine while maintaining the original scope and time context

Focus on questions that would likely find more relevant chunks in the database. Consider:
- What specific financial metrics or data points might be missing?
- What related topics could provide additional context?
- How could the question be rephrased to find more relevant information?

IMPORTANT: Respond ONLY with a valid JSON array. No additional text or formatting.

["What specific metrics were mentioned for the last three quarters?", "Are there any risks discussed in recent quarters?", "What guidance was provided in each quarter?"]"""

