#!/usr/bin/env python3
"""
RAG Analytics Dashboard
Simple script to view analytics data and insights
"""

import asyncio
import asyncpg
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def get_analytics_summary(days=7):
    """Get analytics summary for the last N days"""
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL not found")
        return None
    
    try:
        conn = await asyncpg.connect(db_url)
        
        result = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_queries,
                COUNT(*) FILTER (WHERE overall_success = true) as successful_queries,
                COUNT(*) FILTER (WHERE overall_success = false) as failed_queries,
                ROUND(AVG(total_pipeline_time_ms), 2) as avg_pipeline_time_ms,
                ROUND(AVG(question_analyzer_response_time_ms), 2) as avg_analyzer_time_ms,
                ROUND(AVG(retrieval_time_ms), 2) as avg_retrieval_time_ms,
                ROUND(AVG(llm_response_time_ms), 2) as avg_llm_time_ms,
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(DISTINCT extracted_ticker) as unique_tickers,
                COUNT(*) FILTER (WHERE question_analyzer_retry_count > 0) as analyzer_retries,
                COUNT(*) FILTER (WHERE llm_retry_count > 0) as llm_retries,
                COUNT(*) FILTER (WHERE error_stage = 'question_analyzer') as qa_failures,
                COUNT(*) FILTER (WHERE error_stage = 'chunk_retrieval') as retrieval_failures,
                COUNT(*) FILTER (WHERE error_stage = 'llm_generation') as llm_failures
            FROM rag_analytics 
            WHERE timestamp >= NOW() - INTERVAL '%s days'
        """, days)
        
        await conn.close()
        return dict(result) if result else {}
        
    except Exception as e:
        print(f"❌ Failed to get analytics summary: {e}")
        return {}

async def get_top_questions(days=7, limit=10):
    """Get most common questions"""
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return []
    
    try:
        conn = await asyncpg.connect(db_url)
        
        results = await conn.fetch("""
            SELECT 
                original_question,
                COUNT(*) as question_count,
                COUNT(*) FILTER (WHERE overall_success = true) as success_count,
                ROUND(AVG(total_pipeline_time_ms), 2) as avg_time_ms
            FROM rag_analytics 
            WHERE timestamp >= NOW() - INTERVAL '%s days'
            GROUP BY original_question
            ORDER BY question_count DESC
            LIMIT %s
        """, days, limit)
        
        await conn.close()
        return [dict(row) for row in results]
        
    except Exception as e:
        print(f"❌ Failed to get top questions: {e}")
        return []

async def get_ticker_stats(days=7):
    """Get ticker usage statistics"""
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return []
    
    try:
        conn = await asyncpg.connect(db_url)
        
        results = await conn.fetch("""
            SELECT 
                extracted_ticker,
                COUNT(*) as query_count,
                COUNT(*) FILTER (WHERE overall_success = true) as success_count,
                ROUND(AVG(total_pipeline_time_ms), 2) as avg_time_ms
            FROM rag_analytics 
            WHERE timestamp >= NOW() - INTERVAL '%s days'
            AND extracted_ticker IS NOT NULL
            GROUP BY extracted_ticker
            ORDER BY query_count DESC
            LIMIT 20
        """, days)
        
        await conn.close()
        return [dict(row) for row in results]
        
    except Exception as e:
        print(f"❌ Failed to get ticker stats: {e}")
        return []

async def get_error_analysis(days=7):
    """Get error analysis"""
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return []
    
    try:
        conn = await asyncpg.connect(db_url)
        
        results = await conn.fetch("""
            SELECT 
                error_stage,
                COUNT(*) as error_count,
                COUNT(DISTINCT original_question) as unique_questions,
                STRING_AGG(DISTINCT question_analyzer_error, '; ') as sample_errors
            FROM rag_analytics 
            WHERE timestamp >= NOW() - INTERVAL '%s days'
            AND overall_success = false
            GROUP BY error_stage
            ORDER BY error_count DESC
        """, days)
        
        await conn.close()
        return [dict(row) for row in results]
        
    except Exception as e:
        print(f"❌ Failed to get error analysis: {e}")
        return []

def print_dashboard(summary, top_questions, ticker_stats, error_analysis):
    """Print formatted dashboard"""
    
    print("📊 RAG Analytics Dashboard")
    print("=" * 60)
    
    # Summary stats
    print(f"\n📈 Summary (Last 7 Days)")
    print(f"   Total Queries: {summary.get('total_queries', 0)}")
    print(f"   Success Rate: {summary.get('successful_queries', 0)}/{summary.get('total_queries', 0)} ({summary.get('successful_queries', 0)/max(summary.get('total_queries', 1), 1)*100:.1f}%)")
    print(f"   Avg Pipeline Time: {summary.get('avg_pipeline_time_ms', 0)}ms")
    print(f"   Unique Users: {summary.get('unique_users', 0)}")
    print(f"   Unique Tickers: {summary.get('unique_tickers', 0)}")
    
    # Performance breakdown
    print(f"\n⏱️ Performance Breakdown")
    print(f"   Question Analyzer: {summary.get('avg_analyzer_time_ms', 0)}ms")
    print(f"   Chunk Retrieval: {summary.get('avg_retrieval_time_ms', 0)}ms")
    print(f"   LLM Generation: {summary.get('avg_llm_time_ms', 0)}ms")
    
    # Retry stats
    print(f"\n🔄 Retry Statistics")
    print(f"   Analyzer Retries: {summary.get('analyzer_retries', 0)}")
    print(f"   LLM Retries: {summary.get('llm_retries', 0)}")
    
    # Error breakdown
    if error_analysis:
        print(f"\n❌ Error Analysis")
        for error in error_analysis:
            print(f"   {error['error_stage']}: {error['error_count']} failures")
    
    # Top questions
    if top_questions:
        print(f"\n🔍 Top Questions")
        for i, q in enumerate(top_questions[:5], 1):
            success_rate = q['success_count']/max(q['question_count'], 1)*100
            print(f"   {i}. {q['original_question'][:50]}... ({q['question_count']}x, {success_rate:.1f}% success)")
    
    # Top tickers
    if ticker_stats:
        print(f"\n📈 Top Tickers")
        for i, t in enumerate(ticker_stats[:5], 1):
            success_rate = t['success_count']/max(t['query_count'], 1)*100
            print(f"   {i}. {t['extracted_ticker']}: {t['query_count']} queries ({success_rate:.1f}% success)")

async def main():
    """Main dashboard function"""
    
    print("🚀 Loading RAG Analytics Dashboard...")
    
    # Get all analytics data
    summary = await get_analytics_summary(7)
    top_questions = await get_top_questions(7, 10)
    ticker_stats = await get_ticker_stats(7)
    error_analysis = await get_error_analysis(7)
    
    # Print dashboard
    print_dashboard(summary, top_questions, ticker_stats, error_analysis)
    
    print(f"\n💡 Tips:")
    print(f"   - Monitor success rates to identify issues")
    print(f"   - Watch retry counts for API reliability")
    print(f"   - Track popular questions for content optimization")
    print(f"   - Analyze error stages for debugging")

if __name__ == "__main__":
    asyncio.run(main())
