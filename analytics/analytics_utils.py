"""
Comprehensive RAG Analytics Logger
Captures detailed analytics throughout the entire RAG pipeline
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncpg
import logging

logger = logging.getLogger(__name__)

class RAGAnalyticsLogger:
    """Comprehensive analytics logger for RAG pipeline"""
    
    def __init__(self, db_connection_string: str):
        self.db_connection_string = db_connection_string
        self.session_id = str(uuid.uuid4())
        self.pipeline_start_time = None
        self.analytics_data = {}
        
    async def start_pipeline(self, original_question: str, user_id: str = None, conversation_id: str = None):
        """Start tracking a new RAG pipeline execution"""
        self.pipeline_start_time = time.time()
        self.analytics_data = {
            'session_id': self.session_id,
            'conversation_id': conversation_id,
            'user_id': user_id,
            'original_question': original_question,
            'timestamp': datetime.now(),
            'overall_success': False,
            'error_stage': None
        }
        logger.info(f"🚀 Started RAG analytics tracking for session: {self.session_id}")
    
    async def log_question_analyzer_result(self, 
                                         success: bool, 
                                         analysis_result: Dict[str, Any] = None,
                                         error: str = None,
                                         retry_count: int = 0,
                                         response_time_ms: int = None):
        """Log question analyzer stage results"""
        self.analytics_data.update({
            'question_analyzer_success': success,
            'question_analyzer_error': error,
            'question_analyzer_retry_count': retry_count,
            'question_analyzer_response_time_ms': response_time_ms
        })
        
        if success and analysis_result:
            self.analytics_data.update({
                'is_valid': analysis_result.get('is_valid'),
                'question_type': analysis_result.get('question_type'),
                'extracted_ticker': analysis_result.get('extracted_ticker'),
                'extracted_tickers': analysis_result.get('extracted_tickers', []),
                'rephrased_question': analysis_result.get('rephrased_question'),
                'confidence': analysis_result.get('confidence'),
                'quarter_context': analysis_result.get('quarter_context'),
                'quarter_count': analysis_result.get('quarter_count'),
                'quarter_reference': analysis_result.get('quarter_reference')
            })
        else:
            self.analytics_data['error_stage'] = 'question_analyzer'
            
        logger.info(f"📊 Question analyzer logged: success={success}, retries={retry_count}")
    
    async def log_chunk_retrieval_result(self,
                                       chunks_retrieved: int,
                                       chunks_used: int,
                                       chunk_details: List[Dict[str, Any]] = None,
                                       similarity_threshold: float = None,
                                       chunks_per_quarter: int = None,
                                       retrieval_time_ms: int = None):
        """Log chunk retrieval stage results"""
        self.analytics_data.update({
            'chunks_retrieved': chunks_retrieved,
            'chunks_used': chunks_used,
            'similarity_threshold': similarity_threshold,
            'chunks_per_quarter': chunks_per_quarter,
            'retrieval_time_ms': retrieval_time_ms,
            'chunk_details': json.dumps(chunk_details) if chunk_details else None
        })
        
        logger.info(f"📊 Chunk retrieval logged: {chunks_retrieved} retrieved, {chunks_used} used")
    
    async def log_llm_generation_result(self,
                                      success: bool,
                                      final_answer: str = None,
                                      error: str = None,
                                      retry_count: int = 0,
                                      response_time_ms: int = None,
                                      tokens_used: int = None,
                                      model_used: str = None):
        """Log LLM generation stage results"""
        self.analytics_data.update({
            'llm_success': success,
            'llm_error': error,
            'llm_retry_count': retry_count,
            'llm_response_time_ms': response_time_ms,
            'tokens_used': tokens_used,
            'model_used': model_used,
            'final_answer': final_answer,
            'answer_length': len(final_answer) if final_answer else None
        })
        
        if not success:
            self.analytics_data['error_stage'] = 'llm_generation'
            
        logger.info(f"📊 LLM generation logged: success={success}, retries={retry_count}")
    
    async def finish_pipeline(self, overall_success: bool = None):
        """Finish tracking and save to database"""
        if self.pipeline_start_time:
            total_time_ms = int((time.time() - self.pipeline_start_time) * 1000)
            self.analytics_data['total_pipeline_time_ms'] = total_time_ms
            
        if overall_success is not None:
            self.analytics_data['overall_success'] = overall_success
            
        # Determine overall success if not explicitly set
        if self.analytics_data.get('overall_success') is None:
            qa_success = self.analytics_data.get('question_analyzer_success', False)
            llm_success = self.analytics_data.get('llm_success', False)
            self.analytics_data['overall_success'] = qa_success and llm_success
            
        # Save to database
        await self._save_to_database()
        
        logger.info(f"✅ RAG analytics completed for session: {self.session_id}, success: {self.analytics_data['overall_success']}")
    
    async def _save_to_database(self):
        """Save analytics data to PostgreSQL database"""
        try:
            conn = await asyncpg.connect(self.db_connection_string)
            
            # Prepare the data for insertion
            insert_data = {
                'session_id': self.analytics_data.get('session_id'),
                'conversation_id': self.analytics_data.get('conversation_id'),
                'user_id': self.analytics_data.get('user_id'),
                'timestamp': self.analytics_data.get('timestamp'),
                'original_question': self.analytics_data.get('original_question'),
                'question_analyzer_success': self.analytics_data.get('question_analyzer_success'),
                'question_analyzer_error': self.analytics_data.get('question_analyzer_error'),
                'question_analyzer_retry_count': self.analytics_data.get('question_analyzer_retry_count', 0),
                'question_analyzer_response_time_ms': self.analytics_data.get('question_analyzer_response_time_ms'),
                'is_valid': self.analytics_data.get('is_valid'),
                'question_type': self.analytics_data.get('question_type'),
                'extracted_ticker': self.analytics_data.get('extracted_ticker'),
                'extracted_tickers': self.analytics_data.get('extracted_tickers'),
                'rephrased_question': self.analytics_data.get('rephrased_question'),
                'confidence': self.analytics_data.get('confidence'),
                'quarter_context': self.analytics_data.get('quarter_context'),
                'quarter_count': self.analytics_data.get('quarter_count'),
                'quarter_reference': self.analytics_data.get('quarter_reference'),
                'chunks_retrieved': self.analytics_data.get('chunks_retrieved', 0),
                'chunks_used': self.analytics_data.get('chunks_used', 0),
                'similarity_threshold': self.analytics_data.get('similarity_threshold'),
                'chunks_per_quarter': self.analytics_data.get('chunks_per_quarter'),
                'retrieval_time_ms': self.analytics_data.get('retrieval_time_ms'),
                'chunk_details': self.analytics_data.get('chunk_details'),
                'llm_success': self.analytics_data.get('llm_success'),
                'llm_error': self.analytics_data.get('llm_error'),
                'llm_retry_count': self.analytics_data.get('llm_retry_count', 0),
                'llm_response_time_ms': self.analytics_data.get('llm_response_time_ms'),
                'tokens_used': self.analytics_data.get('tokens_used'),
                'model_used': self.analytics_data.get('model_used'),
                'final_answer': self.analytics_data.get('final_answer'),
                'answer_length': self.analytics_data.get('answer_length'),
                'total_pipeline_time_ms': self.analytics_data.get('total_pipeline_time_ms'),
                'overall_success': self.analytics_data.get('overall_success'),
                'error_stage': self.analytics_data.get('error_stage')
            }
            
            # Insert the record
            await conn.execute("""
                INSERT INTO rag_analytics (
                    session_id, conversation_id, user_id, timestamp, original_question,
                    question_analyzer_success, question_analyzer_error, question_analyzer_retry_count,
                    question_analyzer_response_time_ms, is_valid, question_type, extracted_ticker,
                    extracted_tickers, rephrased_question, confidence, quarter_context,
                    quarter_count, quarter_reference, chunks_retrieved, chunks_used,
                    similarity_threshold, chunks_per_quarter, retrieval_time_ms, chunk_details,
                    llm_success, llm_error, llm_retry_count, llm_response_time_ms,
                    tokens_used, model_used, final_answer, answer_length,
                    total_pipeline_time_ms, overall_success, error_stage
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16,
                    $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30,
                    $31, $32, $33, $34, $35
                )
            """, *insert_data.values())
            
            await conn.close()
            logger.info("💾 Analytics data saved to database successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to save analytics data: {e}")
            # Don't raise the exception - analytics failure shouldn't break the main flow


async def get_analytics_summary(db_connection_string: str, days: int = 7) -> Dict[str, Any]:
    """Get analytics summary for the last N days"""
    try:
        conn = await asyncpg.connect(db_connection_string)
        
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
        logger.error(f"❌ Failed to get analytics summary: {e}")
        return {}


async def log_chat_analytics(
    db,
    ip_address: str,
    user_type,
    query_text: str,
    comprehensive_search: bool,
    success: bool,
    response_time_ms: float,
    citations_count: int,
    user_agent: str = None,
    session_id: str = None,
    error_message: str = None
):
    """Log chat analytics data to the database"""
    try:
        # This is a placeholder implementation
        # You may need to adjust based on your actual database schema
        logger.info(f"📊 Logging chat analytics: user_type={user_type}, success={success}, response_time={response_time_ms}ms")
        
        # If you have a specific analytics table for chat, implement the database logging here
        # For now, we'll just log to the application logger
        
    except Exception as e:
        logger.error(f"❌ Failed to log chat analytics: {e}")


async def get_analytics_data(
    db,
    start_date,
    end_date,
    user_type=None,
    ip_address=None,
    success_only=False,
    limit=1000
):
    """Get analytics data from the database"""
    try:
        # This is a placeholder implementation
        # You may need to adjust based on your actual database schema and requirements
        logger.info(f"📊 Getting analytics data: start_date={start_date}, end_date={end_date}, user_type={user_type}")
        
        # If you have a specific analytics table, implement the database query here
        # For now, we'll return empty data
        return {
            "data": [],
            "total_count": 0,
            "success_count": 0,
            "failure_count": 0
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get analytics data: {e}")
        return {
            "data": [],
            "total_count": 0,
            "success_count": 0,
            "failure_count": 0
        }
