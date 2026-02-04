"""
LLM Utilities - Retry logic and error handling for LLM API calls.

This module provides:
1. Retry decorator with exponential backoff for transient errors
2. User-friendly error messages (no internal details exposed)
3. Unified handling for OpenAI, Cerebras, and other providers
"""

import time
import logging
import functools
from typing import Callable, Any, Optional, List, Type

logger = logging.getLogger(__name__)
rag_logger = logging.getLogger('rag_system')

# Error types that should trigger retries
RETRYABLE_ERRORS = [
    "too_many_requests",
    "rate_limit",
    "overloaded",
    "503",
    "502",
    "500",
    "timeout",
    "connection",
    "queue_exceeded",
    "high traffic",
    "capacity",
    "temporarily unavailable",
]

# User-friendly error messages
USER_FRIENDLY_ERRORS = {
    "rate_limit": "The AI service is currently busy. Please try again in a moment.",
    "timeout": "The request took too long. Please try again.",
    "connection": "Unable to connect to the AI service. Please check your connection and try again.",
    "default": "We encountered an issue processing your request. Please try again.",
    "quota": "Service quota exceeded. Please try again later.",
    "auth": "Authentication error. Please contact support.",
}


class LLMError(Exception):
    """Custom exception for LLM-related errors with user-friendly messages."""

    def __init__(self, user_message: str, technical_message: str = None, retryable: bool = False):
        self.user_message = user_message
        self.technical_message = technical_message or user_message
        self.retryable = retryable
        super().__init__(user_message)

    def __str__(self):
        return self.user_message


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable based on error message/type."""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    for retryable_pattern in RETRYABLE_ERRORS:
        if retryable_pattern in error_str or retryable_pattern in error_type:
            return True

    # Check for specific HTTP status codes in the error
    if hasattr(error, 'status_code'):
        if error.status_code in [429, 500, 502, 503, 504]:
            return True

    return False


def get_user_friendly_message(error: Exception) -> str:
    """Convert technical error to user-friendly message."""
    error_str = str(error).lower()

    if "rate" in error_str or "limit" in error_str or "too_many" in error_str or "queue" in error_str or "traffic" in error_str:
        return USER_FRIENDLY_ERRORS["rate_limit"]
    elif "timeout" in error_str or "timed out" in error_str:
        return USER_FRIENDLY_ERRORS["timeout"]
    elif "connection" in error_str or "connect" in error_str:
        return USER_FRIENDLY_ERRORS["connection"]
    elif "quota" in error_str or "billing" in error_str:
        return USER_FRIENDLY_ERRORS["quota"]
    elif "auth" in error_str or "api_key" in error_str or "unauthorized" in error_str:
        return USER_FRIENDLY_ERRORS["auth"]
    else:
        return USER_FRIENDLY_ERRORS["default"]


def retry_llm_call(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: Optional[List[Type[Exception]]] = None
):
    """
    Decorator for retrying LLM API calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        backoff_factor: Multiplier for delay after each retry
        retryable_exceptions: List of exception types to retry (default: all exceptions)

    Usage:
        @retry_llm_call(max_retries=3)
        def make_api_call():
            return client.chat.completions.create(...)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Check if we should retry
                    should_retry = is_retryable_error(e)
                    if retryable_exceptions:
                        should_retry = should_retry or isinstance(e, tuple(retryable_exceptions))

                    if attempt < max_retries and should_retry:
                        rag_logger.warning(
                            f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)[:100]}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        # No more retries or non-retryable error
                        break

            # All retries exhausted or non-retryable error
            user_message = get_user_friendly_message(last_exception)
            rag_logger.error(f"LLM call failed after {max_retries + 1} attempts: {str(last_exception)}")
            raise LLMError(
                user_message=user_message,
                technical_message=str(last_exception),
                retryable=is_retryable_error(last_exception)
            )

        return wrapper
    return decorator


async def retry_llm_call_async(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
) -> Any:
    """
    Async version of retry logic for LLM API calls.

    Usage:
        result = await retry_llm_call_async(
            lambda: client.chat.completions.create(...),
            max_retries=3
        )
    """
    import asyncio

    last_exception = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            result = func()
            # Handle both sync and async callables
            if asyncio.iscoroutine(result):
                return await result
            return result
        except Exception as e:
            last_exception = e

            if attempt < max_retries and is_retryable_error(e):
                rag_logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)[:100]}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                break

    user_message = get_user_friendly_message(last_exception)
    rag_logger.error(f"LLM call failed after {max_retries + 1} attempts: {str(last_exception)}")
    raise LLMError(
        user_message=user_message,
        technical_message=str(last_exception),
        retryable=is_retryable_error(last_exception)
    )


def safe_llm_call(
    func: Callable,
    fallback_response: str = None,
    max_retries: int = 3,
    initial_delay: float = 1.0,
) -> Any:
    """
    Execute an LLM call with retry logic, returning fallback on failure.

    This is useful when you want a safe call that won't raise exceptions
    but instead returns a fallback value.

    Args:
        func: Callable that makes the LLM API call
        fallback_response: Response to return if all retries fail
        max_retries: Maximum retry attempts
        initial_delay: Initial delay between retries

    Returns:
        API response or fallback_response on failure
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e

            if attempt < max_retries and is_retryable_error(e):
                rag_logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)[:100]}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
            else:
                break

    rag_logger.error(f"LLM call failed, using fallback: {str(last_exception)}")

    if fallback_response is not None:
        return fallback_response

    raise LLMError(
        user_message=get_user_friendly_message(last_exception),
        technical_message=str(last_exception),
        retryable=is_retryable_error(last_exception)
    )


def format_error_for_user(error: Exception) -> str:
    """
    Format any error into a user-friendly message.

    Use this at the top-level error handlers to ensure users
    never see technical error details.
    """
    if isinstance(error, LLMError):
        return error.user_message
    return get_user_friendly_message(error)
