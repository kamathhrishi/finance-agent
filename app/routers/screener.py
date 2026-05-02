"""
Screener router for financial data query endpoints
Handles all query-related functionality including streaming, sorting, pagination, and parallel processing
"""

import asyncio
import json
import time
import uuid
from datetime import date, datetime
from typing import Optional, Dict, Any

import asyncpg
from fastapi import APIRouter, HTTPException, Query, Request as FastAPIRequest, Depends, status
from fastapi.responses import StreamingResponse

# Local imports
from app.auth.auth_utils import get_screener_user
from db.db_utils import get_db
from app.utils import rate_limiter, RATE_LIMIT_PER_MONTH, ADMIN_RATE_LIMIT_PER_MONTH, record_successful_query_usage
from app.utils.logging_utils import log_info, log_error, log_warning
from app.utils import create_error_response, raise_sanitized_http_exception
from app.schemas import (
    QueryRequest, QueryResponse, SortRequest, SortResponse,
    ExpandValueRequest, ExpandValueResponse, ExpandScreenValueRequest,
    PaginationRequest
)

# Import analyzer from the main server
try:
    from agent.screener import FinancialDataAnalyzer, SystemInitializationError, QueryProcessingError
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    FinancialDataAnalyzer = None
    SystemInitializationError = Exception
    QueryProcessingError = Exception

# Global analyzer instance (will be set by main server)
analyzer_instance = None
qualitative_screener_instance = None

# Global dictionary to track active screening requests
active_screening_requests = {}

def set_analyzer_instance(analyzer):
    """Set the analyzer instance from the main server"""
    global analyzer_instance
    analyzer_instance = analyzer

def set_qualitative_screener_instance(screener):
    """Set the qualitative screener instance from the main server"""
    global qualitative_screener_instance
    qualitative_screener_instance = screener

def get_qualitative_screener():
    """Get the qualitative screener instance (may be None)."""
    return qualitative_screener_instance

def get_analyzer() -> FinancialDataAnalyzer:
    """Get the analyzer instance"""
    if not ANALYZER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Query features disabled - analyzer not available")
    if analyzer_instance is None:
        raise HTTPException(status_code=503, detail="Financial Analyzer is not available")
    return analyzer_instance

# Create router
router = APIRouter(prefix="/screener", tags=["screener"])

@router.get("/smart/stream")
@router.get("/query/stream")
async def stream_query(
    request: FastAPIRequest,
    question: str = Query(..., min_length=1, max_length=1000, description="Natural language question about financial data (max 1000 characters)"),
    page: int = Query(1, ge=1, description="Page number for pagination"),
    page_size: Optional[int] = Query(None, ge=1, le=1000, description="Number of records per page"),
    top_n: Optional[int] = Query(None, ge=1, le=500, description="Max companies to return for qualitative screening"),
    current_user: dict = Depends(get_screener_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Execute a financial data query and stream reasoning events with rate limiting."""
    user_id = current_user["id"]
    is_admin = current_user.get("is_admin", False)
    is_guest = current_user.get("is_guest", False)

    # UNIFIED RATE LIMIT CHECK - both minute and monthly limits together
    try:
        # Check both minute and monthly limits together
        allowed, limit_info = await rate_limiter.check_rate_limit_with_monthly(user_id, is_admin, db) if not is_guest else (True, {})
        
        if not allowed:
            # Rate limit exceeded (either minute or monthly)
            log_warning(f"Rate limit exceeded for user {user_id}: {limit_info['message']}")
            
            async def rate_limit_error_generator():
                error_event = {
                    'type': 'error',
                    'message': limit_info['message'],
                    'error': 'RATE_LIMIT_EXCEEDED',
                    'rate_limit_info': {
                        'limit_type': limit_info['limit_type'],
                        'limit': limit_info['limit'],
                        'reset_time': limit_info['reset_time']
                    }
                }
                yield f"data: {json.dumps(error_event)}\n\n"
            
            headers = {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
            return StreamingResponse(rate_limit_error_generator(), headers=headers)
        
        # Record the request for rate limiting purposes only (not for billing)
        rate_limiter.record_request(user_id)
        
    except Exception as e:
        log_error(f"Rate limit check failed for user {user_id}: {e}")
        # If rate limit check fails, be conservative and block
        async def rate_limit_error_generator():
            error_event = {
                'type': 'error',
                'message': 'Rate limit check failed. Please try again later.',
                'error': 'RATE_LIMIT_EXCEEDED',
                'rate_limit_info': {
                    'limit_type': 'month',
                    'limit': RATE_LIMIT_PER_MONTH,
                    'reset_time': 'unknown'
                }
            }
            yield f"data: {json.dumps(error_event)}\n\n"
        
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(rate_limit_error_generator(), headers=headers)
    
    # Ensure at least one screener is available
    if qualitative_screener_instance is None:
        analyzer = get_analyzer()  # raises 503 if unavailable
    else:
        try:
            analyzer = get_analyzer()
        except HTTPException:
            analyzer = None

    # Track this request for potential cancellation
    request_id = str(uuid.uuid4())
    active_screening_requests[user_id] = {
        "request_id": request_id,
        "started_at": datetime.utcnow(),
        "question": question,
        "cancelled": False
    }

    log_info(f"Starting stream query for user {current_user['id']}: '{question}'")
    
    async def event_generator():
        query_successful = False
        try:
            log_info(f"User {current_user['id']} starting stream for: '{question}'")

            # Route through QualitativeScreener if available (handles both financial and qualitative)
            if qualitative_screener_instance is not None:
                stream = qualitative_screener_instance.screen_with_streaming(
                    question=question,
                    top_n=top_n or page_size,
                    page=page,
                    page_size=page_size,
                )
                async for event in stream:
                    if user_id in active_screening_requests and active_screening_requests[user_id].get("cancelled", False):
                        log_info(f"Request {request_id} was cancelled, stopping stream")
                        break
                    if await request.is_disconnected():
                        log_warning(f"Client disconnected for user {current_user['id']}. Stopping stream.")
                        break
                    if event.get('type') == 'result' and not event.get('error'):
                        query_successful = True
                    yield f"data: {json.dumps(event)}\n\n"
                    await asyncio.sleep(0.02)
            else:
                # Fallback: financial-only screener
                for event in analyzer.query_with_streaming(
                    question=question,
                    page=page,
                    page_size=page_size
                ):
                    if user_id in active_screening_requests and active_screening_requests[user_id].get("cancelled", False):
                        log_info(f"Request {request_id} was cancelled, stopping stream")
                        break
                    if await request.is_disconnected():
                        log_warning(f"Client disconnected for user {current_user['id']}. Stopping stream.")
                        break
                    if event.get('type') == 'result' and not event.get('error'):
                        query_successful = True
                    yield f"data: {json.dumps(event)}\n\n"
                    await asyncio.sleep(0.02)
            
            # Record usage only if query was successful and user is authenticated
            if query_successful and not is_guest:
                await record_successful_query_usage(current_user['id'], db, 0.02)  # COST_PER_REQUEST from main server
        
        except Exception as e:
            log_error(f"Error during query streaming for user {current_user['id']}: {e}", exc_info=True)
            error_event = {
                'type': 'error',
                'message': f'An error occurred during processing.',
                'error': str(e)
            }
            yield f"data: {json.dumps(error_event)}\n\n"
        finally:
            # Clean up the request tracking
            if user_id in active_screening_requests:
                del active_screening_requests[user_id]

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # For NGINX proxy buffering
    }
    return StreamingResponse(event_generator(), headers=headers)

@router.post("/query/sort", response_model=SortResponse)
async def sort_query_results(
    sort_request: SortRequest,
    request: FastAPIRequest,
    current_user: dict = Depends(get_screener_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """
    Sort cached query results on the server side
    """
    if not ANALYZER_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Sorting functionality is not available. The financial analyzer is not initialized."
        )
    
    analyzer = get_analyzer()
    
    is_guest = current_user.get("is_guest", False)

    try:
        # For now, we'll need the original question to find cached data
        # In a production system, you might want to store query_id -> question mapping

        if sort_request.question and (is_guest or not sort_request.query_id):
            # Use question directly when provided (guest users or when no query_id)
            question = sort_request.question
            log_info(f"🔍 SCREENER SORT: Using provided question: '{question}'")
        elif not sort_request.query_id:
            # Get the most recent query from this user
            recent_query = await db.fetchrow('''
                SELECT question FROM query_history
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT 1
            ''', uuid.UUID(current_user["id"]))

            if not recent_query:
                raise HTTPException(status_code=404, detail="No recent query found to sort")

            question = recent_query['question']
            log_info(f"🔍 SCREENER SORT: Using most recent question from DB: '{question}'")
        else:
            # Get question from query_id
            query_record = await db.fetchrow('''
                SELECT question FROM query_history
                WHERE id = $1 AND user_id = $2
            ''', uuid.UUID(sort_request.query_id), uuid.UUID(current_user["id"]))

            if not query_record:
                raise HTTPException(status_code=404, detail="Query not found or access denied")

            question = query_record['question']
            log_info(f"🔍 SCREENER SORT: Using question from query_id {sort_request.query_id}: '{question}'")
        
        log_info(f"🔍 SCREENER SORT: About to call sort_and_paginate_cached_data with:")
        log_info(f"  - Question: '{question}'")
        log_info(f"  - Column: '{sort_request.column}'")
        log_info(f"  - Direction: '{sort_request.direction}'")
        
        # Sort and paginate the cached data
        result = analyzer.sort_and_paginate_cached_data(
            question=question,
            column=sort_request.column,
            direction=sort_request.direction,
            page=1,  # Always start from page 1 when sorting
            page_size=20  # Default page size
        )
        
        log_info(f"🔍 SCREENER SORT: Result success: {result['success']}")
        if not result['success']:
            log_info(f"🔍 SCREENER SORT: Error: {result.get('error')}")
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result.get('error', 'Sorting failed'))
        
        return SortResponse(
            success=True,
            data_rows=result['data_rows'],
            columns=result['columns'],
            friendly_columns=result['friendly_columns'],
            message=result['message'],
            sort_applied=result['sort_applied'],
            total_rows=result['total_rows'],
            pagination_info=result.get('pagination_info')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_response = create_error_response(e, "server-side sorting", current_user.get("id"))
        log_error(f"❌ Server-side sorting failed")
        
        return SortResponse(
            success=False,
            data_rows=[],
            columns=[],
            friendly_columns={},
            message=error_response["error"],
            sort_applied={'column': sort_request.column, 'direction': sort_request.direction},
            total_rows=0
        )

@router.post("/query/complete-dataset", response_model=QueryResponse)
async def get_complete_dataset_for_screen(
    query_request: QueryRequest,
    request: FastAPIRequest,
    current_user: dict = Depends(get_screener_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Get complete dataset for screen saving (bypasses page_size limits)"""

    # SIMPLE RATE LIMIT CHECK - directly in the function
    user_id = current_user["id"]
    is_admin = current_user.get("is_admin", False)
    is_guest = current_user.get("is_guest", False)

    try:
        if not is_guest:
            # Check minute limit (in-memory only)
            rate_limiter.check_rate_limit(user_id, is_admin)

            # Check monthly limit from database (persistent)
            today = date.today()
            month_start = today.replace(day=1)
            monthly_limit = ADMIN_RATE_LIMIT_PER_MONTH if is_admin else RATE_LIMIT_PER_MONTH

            # Get monthly count from database
            monthly_count = await db.fetchval('''
                SELECT COALESCE(SUM(request_count), 0)
                FROM user_usage
                WHERE user_id = $1 AND request_date >= $2
            ''', uuid.UUID(user_id), month_start)
        
            # Check if monthly limit exceeded
            if monthly_count >= monthly_limit:
                if today.month == 12:
                    next_month = today.replace(year=today.year + 1, month=1, day=1)
                else:
                    next_month = today.replace(month=today.month + 1, day=1)

                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Monthly limit exceeded. Maximum {monthly_limit} requests per month.",
                    headers={
                        "X-RateLimit-Limit": str(monthly_limit),
                        "X-RateLimit-Reset": next_month.isoformat(),
                        "X-RateLimit-Remaining": "0"
                    }
                )

            # Record the request for rate limiting purposes only (not for billing)
            rate_limiter.record_request(user_id)

    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Rate limit check failed for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit check failed. Please try again later."
        )

    try:
        start_time = time.time()
        
        # Get analyzer instance
        analyzer = get_analyzer()
        
        # Execute query with no pagination to get complete dataset
        result = analyzer.query(query_request.question, page=1, page_size=None)
        
        execution_time = time.time() - start_time
        
        if result.get('error'):
            return QueryResponse(
                success=False,
                execution_time=execution_time,
                error=result.get('error'),
                message="Failed to retrieve complete dataset"
            )
        
        # Single-sheet response
        response_data = QueryResponse(
            success=not bool(result.get('error')),
            execution_time=execution_time,
            sql_query_generated=result.get('sql_query_generated'),
            columns=result.get('columns'),
            friendly_columns=result.get('friendly_columns'),
            data_rows=result.get('data_rows'),
            message=result.get('message'),
            error=result.get('error'),
            tables_used=result.get('tables_used'),
            pagination_info=None,  # No pagination for complete dataset
            used_stored_results=result.get('used_stored_results', False),
            used_pipeline_cache=result.get('cache_type_hit') == "pipeline_cache" if result.get('cache_type_hit') else False
        )
        
        # Record usage for successful queries (not for guests)
        if not bool(result.get('error')) and not is_guest:
            await record_successful_query_usage(current_user["id"], db, 0.02)  # COST_PER_REQUEST from main server

        return response_data
        
    except Exception as e:
        error_response = create_error_response(e, "complete dataset retrieval", current_user.get("id"))
        return QueryResponse(
            success=False,
            execution_time=0,
            error=error_response["error"],
            message=error_response["message"]
        )

@router.post("/query/expand-value", response_model=ExpandValueResponse)
async def expand_truncated_value(
    expand_request: ExpandValueRequest,
    request: FastAPIRequest,
    current_user: dict = Depends(get_screener_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """
    Get the full value for a truncated cell from cached data
    """
    if not ANALYZER_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Expand functionality is not available. The financial analyzer is not initialized."
        )
    
    analyzer = get_analyzer()
    
    try:
        # Find cached data using fuzzy matching
        cached_pipeline = analyzer.find_cached_pipeline_by_question_fuzzy(expand_request.question)
        
        if not cached_pipeline:
            return ExpandValueResponse(
                success=False,
                error="No cached data found for this question"
            )
        
        # Get the raw dataframe
        if 'raw_results_df' not in cached_pipeline:
            return ExpandValueResponse(
                success=False,
                error="No raw results data found in cache"
            )
        
        df = cached_pipeline['raw_results_df']
        
        # Validate row index
        if expand_request.row_index >= len(df):
            return ExpandValueResponse(
                success=False,
                error=f"Row index {expand_request.row_index} is out of bounds. Data has {len(df)} rows."
            )
        
        # Validate column exists
        if expand_request.column_name not in df.columns:
            available_cols = list(df.columns)
            return ExpandValueResponse(
                success=False,
                error=f"Column '{expand_request.column_name}' not found. Available columns: {', '.join(available_cols)}"
            )
        
        # Get the full value
        full_value = df.iloc[expand_request.row_index][expand_request.column_name]
        
        # Convert to string and handle None/NaN values
        if full_value is None or pd.isna(full_value):
            full_value = "—"
        else:
            full_value = str(full_value)
        
        return ExpandValueResponse(
            success=True,
            full_value=full_value,
            message=f"Retrieved full value for {expand_request.column_name} at row {expand_request.row_index}"
        )
        
    except Exception as e:
        log_error(f"❌ Error expanding truncated value: {e}")
        error_response = create_error_response(e, "value expansion", current_user.get("id"))
        return ExpandValueResponse(
            success=False,
            error=error_response["error"]
        )

@router.post("/query/paginate", response_model=QueryResponse)
async def paginate_cached_results(
    pagination_request: PaginationRequest,
    request: FastAPIRequest,
    current_user: dict = Depends(get_screener_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Paginate cached results without triggering full query processing"""
    # NO RATE LIMITING for pagination - it's just serving cached data
    # Only successful queries that return results should count as requests
    
    if not ANALYZER_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Query functionality is not available. The financial analyzer is not initialized."
        )
    
    analyzer = get_analyzer()
    start_time = time.time()
    
    try:
        # Use the analyzer's cached pagination method (no sorting)
        result = analyzer.paginate_cached_data(
            question=pagination_request.question,
            page=pagination_request.page,
            page_size=pagination_request.page_size
        )
        
        execution_time = time.time() - start_time
        
        if not result.get('success'):
            raise HTTPException(
                status_code=404,
                detail=result.get('error', 'No cached data found for this query')
            )
        
        # Format response to match QueryResponse model
        response_data = QueryResponse(
            success=True,
            execution_time=execution_time,
            query_type="single_sheet",
            sql_query_generated=result.get('sql_query', ''),
            columns=result.get('columns', []),
            friendly_columns=result.get('friendly_columns', {}),
            data_rows=result.get('data_rows', []),
            message=result.get('message', ''),
            error=None,
            tables_used=result.get('tables_used', []),
            pagination_info=result.get('pagination_info'),
            used_stored_results=True,
            used_pipeline_cache=True
        )
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        error_response = create_error_response(e, "pagination", current_user.get("id"))
        error_message = error_response["error"]
        
        return QueryResponse(
            success=False,
            execution_time=execution_time,
            query_type="single_sheet",
            error=error_message,
            message="Pagination failed"
        )

@router.post("/query/parallel/quarters", response_model=QueryResponse)
async def query_multiple_quarters_parallel(
    request: dict,
    http_request: FastAPIRequest,
    current_user: dict = Depends(get_screener_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Query multiple quarters in parallel for comparative analysis"""
    user_id = current_user["id"]
    is_admin = current_user.get("is_admin", False)
    is_guest = current_user.get("is_guest", False)

    # Check rate limits (skip for guests)
    if not is_guest:
        try:
            allowed, limit_info = await rate_limiter.check_rate_limit_with_monthly(user_id, is_admin, db)
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=limit_info['message']
                )
        except HTTPException:
            raise
        except Exception as e:
            log_error(f"Rate limit check failed: {e}")
            raise HTTPException(status_code=500, detail="Rate limit check failed")
    
    if not ANALYZER_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Query functionality is not available. The financial analyzer is not initialized."
        )
    
    # Validate request
    question = request.get("question", "").strip()
    quarters = request.get("quarters", [])
    page = request.get("page", 1)
    page_size = request.get("page_size", 20)
    
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    if not quarters or len(quarters) < 2:
        raise HTTPException(status_code=400, detail="At least 2 quarters are required for parallel processing")
    
    if len(quarters) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 quarters allowed for parallel processing")
    
    analyzer = get_analyzer()
    start_time = time.time()
    
    try:
        # Execute parallel multi-quarter query
        result = analyzer.query_multiple_quarters_parallel(
            question=question,
            quarters=quarters,
            page=page,
            page_size=page_size
        )
        
        execution_time = time.time() - start_time

        # Record successful query usage (skip for guests)
        if not is_guest:
            await record_successful_query_usage(user_id, db, 0.02)  # COST_PER_REQUEST from main server

        # Format response
        response_data = QueryResponse(
            success=True,
            execution_time=execution_time,
            query_type="multi_quarter_parallel",
            sql_query_generated=result.get('sql_query_generated', ''),
            columns=result.get('columns', []),
            friendly_columns=result.get('friendly_columns', {}),
            data_rows=result.get('data_rows', []),
            message=result.get('message', ''),
            error=result.get('error'),
            tables_used=result.get('tables_used', []),
            pagination_info=result.get('pagination_info', {}),
            used_stored_results=result.get('used_stored_results', False),
            used_pipeline_cache=result.get('cache_type_hit') == 'pipeline_cache',
            quarter_results=result.get('quarter_results', {}),
            total_quarters=result.get('total_quarters', 0),
            successful_quarters=result.get('successful_quarters', 0),
            failed_quarters=result.get('failed_quarters', 0)
        )
        
        return response_data
        
    except Exception as e:
        raise_sanitized_http_exception(
            e, 
            "parallel multi-quarter query", 
            current_user.get("id"),
            status_code=500
        )

@router.post("/query/parallel/companies", response_model=QueryResponse)
async def query_multiple_companies_parallel(
    request: dict,
    http_request: FastAPIRequest,
    current_user: dict = Depends(get_screener_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Query multiple companies in parallel for comparative analysis"""
    user_id = current_user["id"]
    is_admin = current_user.get("is_admin", False)
    is_guest = current_user.get("is_guest", False)

    # Check rate limits (skip for guests)
    if not is_guest:
        try:
            allowed, limit_info = await rate_limiter.check_rate_limit_with_monthly(user_id, is_admin, db)
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=limit_info['message']
                )
        except HTTPException:
            raise
        except Exception as e:
            log_error(f"Rate limit check failed: {e}")
            raise HTTPException(status_code=500, detail="Rate limit check failed")
    
    if not ANALYZER_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Query functionality is not available. The financial analyzer is not initialized."
        )
    
    # Validate request
    question = request.get("question", "").strip()
    companies = request.get("companies", [])
    page = request.get("page", 1)
    page_size = request.get("page_size", 20)
    
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    if not companies or len(companies) < 2:
        raise HTTPException(status_code=400, detail="At least 2 companies are required for parallel processing")
    
    if len(companies) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 companies allowed for parallel processing")
    
    analyzer = get_analyzer()
    start_time = time.time()
    
    try:
        # Execute parallel multi-company query
        result = analyzer.query_multiple_companies_parallel(
            question=question,
            companies=companies,
            page=page,
            page_size=page_size
        )
        
        execution_time = time.time() - start_time
        
        # Record successful query usage (skip for guests)
        if not is_guest:
            await record_successful_query_usage(user_id, db, 0.02)  # COST_PER_REQUEST from main server

        # Format response
        response_data = QueryResponse(
            success=True,
            execution_time=execution_time,
            query_type="multi_company_parallel",
            sql_query_generated=result.get('sql_query_generated', ''),
            columns=result.get('columns', []),
            friendly_columns=result.get('friendly_columns', {}),
            data_rows=result.get('data_rows', []),
            message=result.get('message', ''),
            error=result.get('error'),
            tables_used=result.get('tables_used', []),
            pagination_info=result.get('pagination_info', {}),
            used_stored_results=result.get('used_stored_results', False),
            used_pipeline_cache=result.get('cache_type_hit') == 'pipeline_cache',
            company_results=result.get('company_results', {}),
            total_companies=result.get('total_companies', 0),
            successful_companies=result.get('successful_companies', 0),
            failed_companies=result.get('failed_companies', 0)
        )
        
        return response_data
        
    except Exception as e:
        raise_sanitized_http_exception(
            e, 
            "parallel multi-company query", 
            current_user.get("id"),
            status_code=500
        )

@router.post("/cancel")
async def cancel_screening_request(
    request: FastAPIRequest,
    current_user: dict = Depends(get_screener_user)
):
    """Cancel the current active screening request for the user"""
    user_id = current_user.get('id')
    
    try:
        if user_id in active_screening_requests:
            # Mark the request as cancelled
            active_screening_requests[user_id]['cancelled'] = True
            log_info(f"Screening request cancelled for user {user_id}")
            
            return {
                "success": True,
                "message": "Screening request cancelled successfully"
            }
        else:
            return {
                "success": False,
                "message": "No active screening request found"
            }
            
    except Exception as e:
        log_error(f"Error cancelling screening request for user {user_id}: {e}")
        return {
            "success": False,
            "message": f"Failed to cancel request: {str(e)}"
        }
