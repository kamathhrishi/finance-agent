"""
Logfire Configuration for StrataLens AI
Handles observability setup with environment-based configuration
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def configure_logfire(
    service_name: str = "stratalens-ai",
    environment: Optional[str] = None
) -> bool:
    """
    Configure Logfire for observability.
    
    Args:
        service_name: Name of the service (default: stratalens-ai)
        environment: Environment name (default: from ENVIRONMENT env var or 'production')
    
    Returns:
        bool: True if Logfire was configured successfully, False otherwise
    
    Environment Variables:
        LOGFIRE_TOKEN: Write token from logfire.pydantic.dev (required)
        ENVIRONMENT: Environment name (optional, default: production)
        LOGFIRE_ENABLED: Set to 'false' to disable Logfire (optional)
    """
    # Check if Logfire should be enabled
    logfire_enabled = os.getenv("LOGFIRE_ENABLED", "true").lower() == "true"
    if not logfire_enabled:
        logger.info("🔕 Logfire is disabled via LOGFIRE_ENABLED=false")
        return False
    
    # Get Logfire token from .env
    logfire_token = os.getenv("LOGFIRE_TOKEN")
    
    if not logfire_token:
        logger.warning("⚠️ LOGFIRE_TOKEN not found in .env - Logfire disabled")
        logger.warning("   Get your token from https://logfire.pydantic.dev")
        logger.warning("   Add to .env: LOGFIRE_TOKEN=your_token_here")
        return False
    
    try:
        import logfire  # lazy import — fails gracefully if package is broken
        # Determine environment
        env = environment or os.getenv("ENVIRONMENT", "production")

        # Configure Logfire
        logfire.configure(
            token=logfire_token,
            service_name=service_name,
            environment=env,
            # Send console logs to Logfire (optional)
            send_to_logfire=True,
            # Console output control
            console=False,  # Disable Logfire console output (use your existing logging)
        )
        
        logger.info(f"✅ Logfire configured successfully")
        logger.info(f"   Service: {service_name}")
        logger.info(f"   Environment: {env}")
        logger.info(f"   Dashboard: https://logfire.pydantic.dev")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to configure Logfire: {e}")
        logger.error("   Continuing without Logfire observability")
        return False


def instrument_all():
    """
    Auto-instrument common libraries used in StrataLens AI.
    Call this after configure_logfire() returns True.

    IMPORTANT: For OpenAI instrumentation to capture prompts/completions,
    this must be called BEFORE any OpenAI client is instantiated.
    """
    import logfire  # lazy import
    # Enable capturing of message content (prompts/completions) for GenAI instrumentation
    # This is REQUIRED for OpenAI prompts to show in Logfire
    os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")

    try:
        # Instrument asyncpg for database query tracking
        logfire.instrument_asyncpg()
        logger.info("✅ Instrumented asyncpg (database queries)")
    except Exception as e:
        logger.warning(f"⚠️ Could not instrument asyncpg: {e}")

    try:
        # Instrument httpx for HTTP client tracking with full request/response capture
        # capture_all=True captures headers and body for both request and response
        logfire.instrument_httpx(capture_all=True)
        logger.info("✅ Instrumented httpx (HTTP clients with full request/response)")
    except Exception as e:
        logger.warning(f"⚠️ Could not instrument httpx: {e}")

    try:
        # Instrument OpenAI for LLM call tracking (tokens, costs, latency)
        # capture_all=True enables full prompt/completion logging
        # Also requires OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true (set above)
        logfire.instrument_openai(capture_all=True)
        logger.info("✅ Instrumented OpenAI (LLM calls with full prompts/completions)")
    except Exception as e:
        logger.warning(f"⚠️ Could not instrument OpenAI: {e}")

    # Note: Cerebras is NOT auto-instrumented by Logfire
    # Manual spans are added in response_generator.py and question_analyzer.py
    # to capture Cerebras LLM prompts and completions
    logger.info("ℹ️ Cerebras calls tracked via manual Logfire spans in agent code")


# Global flag to track if Logfire is active
LOGFIRE_ACTIVE = False


def init_logfire() -> bool:
    """
    Initialize Logfire with full instrumentation.
    Call this early in your application startup.
    
    Returns:
        bool: True if Logfire is active and ready to use
    """
    global LOGFIRE_ACTIVE
    
    # Configure Logfire
    configured = configure_logfire()
    
    if configured:
        # Instrument libraries
        instrument_all()
        LOGFIRE_ACTIVE = True
        logger.info("🔥 Logfire is active and ready")
    else:
        LOGFIRE_ACTIVE = False
        logger.info("📊 Running without Logfire (using standard logging)")
    
    return LOGFIRE_ACTIVE


def is_logfire_active() -> bool:
    """Check if Logfire is currently active."""
    return LOGFIRE_ACTIVE
