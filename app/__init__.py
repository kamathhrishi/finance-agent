"""
FastAPI Application Factory

This module creates and configures the FastAPI application instance.
"""

from dotenv import load_dotenv
load_dotenv()

# ─── Pre-import bootstrap of fs_research_agent path ──────────────────────────
#
# The chat router instantiates the FS agent at module-import time (line ~120
# of app/routers/chat.py). The agent validates `data_root.is_dir()` and
# raises if it doesn't exist. On Railway the default location
# (/app/fs_research_agent/data) does not exist — the corpus lives on the
# persistent volume at /data/fs_research_corpus. The lifespan sets that env
# var, but lifespan runs AFTER router imports, so the agent fails first.
#
# Fix: set the default + ensure the dir exists BEFORE any router imports run.
# Idempotent — only acts when the env var is unset and we're on Railway.
import os as _os
if not _os.getenv("FS_RESEARCH_DATA_ROOT"):
    _on_railway = bool(
        _os.getenv("RAILWAY_ENVIRONMENT")
        or _os.getenv("RAILWAY_PROJECT_ID")
        or _os.getenv("RAILWAY_SERVICE_ID")
    )
    if _on_railway:
        _os.environ["FS_RESEARCH_DATA_ROOT"] = "/data/fs_research_corpus"

# Ensure the data root dir exists (empty is fine — bootstrap fills it later).
# This stops the agent's hard `is_dir()` check from failing at import time.
try:
    from pathlib import Path as _Path
    _root = _os.getenv("FS_RESEARCH_DATA_ROOT") or str(
        _Path(__file__).resolve().parent.parent / "fs_research_agent" / "data"
    )
    _Path(_root).mkdir(parents=True, exist_ok=True)
except Exception:
    pass  # If we can't mkdir (permissions, etc.), agent init will surface it.

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

