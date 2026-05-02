"""
FastAPI Application Factory

This module creates and configures the FastAPI application instance.
"""

from dotenv import load_dotenv
load_dotenv()

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import settings
from app.lifespan import lifespan
from app.middleware import setup_middleware
from app.routes import setup_routes
from app.utils.logfire_config import init_logfire, is_logfire_active

_error_logger = logging.getLogger("app.errors")

# Create FastAPI app
app = FastAPI(
    title=settings.APPLICATION.TITLE,
    description=settings.APPLICATION.DESCRIPTION,
    version=settings.APPLICATION.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.APPLICATION.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.APPLICATION.ENABLE_REDOC else None
)

# Instrument FastAPI with Logfire (must be done before adding routes)
try:
    import logfire
    if is_logfire_active():
        logfire.instrument_fastapi(app)
        from app.utils.logging_utils import log_info
        log_info("✅ FastAPI instrumented with Logfire")
except Exception as e:
    from app.utils.logging_utils import log_warning
    log_warning(f"⚠️ Could not instrument FastAPI with Logfire: {e}")

# ─── Global exception handlers ───────────────────────────────────────────────
#
# Hard rule: clients NEVER see internal exception text, stack traces, library
# names, file paths, or upstream error bodies. They get a vague message + a
# correlation id they can quote when reporting an issue. Operators see the
# full traceback in logs and (if configured) in the logfire trace tied to the
# same correlation id.
#
# Two handlers:
#   1. RequestValidationError → 400 with a generic "invalid request" message.
#      We DO NOT echo the offending field paths/types to the client because
#      they reveal the API schema shape and have leaked field names in the past.
#   2. Generic Exception → 500 with a generic "something went wrong" message.
#      Catches anything not deliberately raised as HTTPException, including
#      KeyError, TypeError, AttributeError, third-party library exceptions, etc.
#
# Deliberately NOT overridden: HTTPException / StarletteHTTPException —
# those carry developer-chosen `detail` strings (e.g. "Invalid date format,
# use YYYY-MM-DD") that ARE user-actionable. Endpoint authors should make
# sure those `detail` strings don't leak internals; this handler doesn't
# second-guess them.


def _correlation_id() -> str:
    return uuid.uuid4().hex[:12]


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    cid = _correlation_id()
    # Log full validation detail server-side for debugging
    _error_logger.warning(
        "request_validation_failed cid=%s path=%s method=%s errors=%s",
        cid, request.url.path, request.method, exc.errors(),
    )
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Invalid request. Please check your input and try again.",
            "correlation_id": cid,
        },
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    # Re-raise HTTPException so FastAPI handles it normally (it inherits from
    # Exception, so this handler would otherwise swallow it). The check via
    # StarletteHTTPException covers both fastapi.HTTPException and starlette's.
    if isinstance(exc, StarletteHTTPException):
        raise exc

    cid = _correlation_id()
    # Full exception + traceback to logs; operators look up by correlation id.
    _error_logger.exception(
        "unhandled_exception cid=%s path=%s method=%s exc_type=%s",
        cid, request.url.path, request.method, type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Something went wrong. Please try again in a moment.",
            "correlation_id": cid,
        },
    )


# Setup middleware
setup_middleware(app)

# Setup routes
setup_routes(app)

# Mount static files directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

