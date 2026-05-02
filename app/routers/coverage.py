"""
Coverage Router

Public read-only endpoints that power the Companies and Latest Filings tabs.

Backed by `fs_research_agent/data/_coverage_index.json` (built by
`fs_research_agent.coverage_index`). Loaded into memory on first request and
hot-reloaded when the file's mtime changes — so a watcher cycle that rewrites
the index is reflected on the next request without a server restart.

Endpoints:
  GET /coverage/companies             — full universe + per-form filing counts
  GET /coverage/companies/{ticker}    — drill-down: every filing for one ticker
  GET /coverage/latest                — newest-first feed across all coverage
  GET /coverage/status                — generated_at + totals (for "Last refreshed" badge)
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from fs_research_agent.coverage_index import index_path, load_index, rebuild

logger = logging.getLogger("app.routers.coverage")

router = APIRouter(prefix="/coverage", tags=["coverage"])


# ─────────────────────────────────────────────────────────────────────────────
# In-memory cache with mtime-based hot reload
# ─────────────────────────────────────────────────────────────────────────────


class _IndexCache:
    """
    Holds the coverage index in memory; reloads from disk if the file's mtime
    has advanced since the last read. Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Optional[Dict[str, Any]] = None
        self._mtime: float = 0.0

    def get(self) -> Dict[str, Any]:
        path: Path = index_path()
        try:
            current_mtime = path.stat().st_mtime
        except FileNotFoundError:
            current_mtime = 0.0

        with self._lock:
            if self._data is not None and current_mtime == self._mtime:
                return self._data

            # Either first load OR file changed → reload (or build if missing)
            data = load_index()
            if data is None:
                logger.info("coverage index missing — building it now")
                data = rebuild()
                try:
                    current_mtime = path.stat().st_mtime
                except FileNotFoundError:
                    current_mtime = 0.0

            self._data = data
            self._mtime = current_mtime
            logger.info(
                f"coverage index loaded: {data['stats']['filing_count']} filings, "
                f"{data['stats']['company_count']} companies"
            )
            return self._data


_cache = _IndexCache()


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic response models
# ─────────────────────────────────────────────────────────────────────────────


class FilingSummary(BaseModel):
    """One filing row — what shows up in tables/lists. filing_date is the SEC
    submission date and is the primary date users see."""
    ticker: str
    company_name: str
    form: str                              # "10-K" / "10-Q" / "8-K"
    fiscal_year_label: Optional[str]       # "FY2025" or None for some 8-Ks
    quarter_label: Optional[str]           # "Q3" for 10-Q, else None
    period_label: str                      # human-readable cell label
    filing_date: str                       # YYYY-MM-DD — when SEC received it
    accession: str
    path: str                              # relative to data_root, e.g. "filings/AAPL/10-K/FY2024"
    section_count: int = 0                 # how many parsed sections
    exhibit_count: int = 0                 # how many keepable exhibits
    filing_chars: int = 0                  # rough size of the main filing markdown


class CompanySummary(BaseModel):
    """One row in the Companies grid."""
    ticker: str
    cik: str
    company_name: str
    counts: Dict[str, int]                 # {"10-K": 5, "10-Q": 12, "8-K": 38}
    total: int
    latest_filing_date: Optional[str]      # newest filing on file for this ticker, or None


class CompanyDetail(BaseModel):
    """Drill-down: one company + every filing for it."""
    ticker: str
    cik: str
    company_name: str
    counts: Dict[str, int]
    total: int
    filings: List[FilingSummary]           # newest-first


class StatusResponse(BaseModel):
    generated_at: str
    company_count: int
    filing_count: int
    by_form: Dict[str, int]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _aggregate_company_stats(idx: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """For each ticker in the universe, compute per-form counts and latest filing date."""
    by_ticker: Dict[str, Dict[str, Any]] = {}
    for c in idx.get("companies", []):
        by_ticker[c["ticker"]] = {
            "ticker": c["ticker"],
            "cik": c.get("cik", ""),
            "company_name": c.get("company_name", ""),
            "counts": {},
            "total": 0,
            "latest_filing_date": None,
        }

    for f in idx.get("filings", []):
        t = f.get("ticker")
        entry = by_ticker.get(t)
        if entry is None:
            # Filing for a ticker not in the universe — surface anyway with empty
            # name so we don't silently drop data
            entry = {
                "ticker": t,
                "cik": "",
                "company_name": f.get("company_name", ""),
                "counts": {},
                "total": 0,
                "latest_filing_date": None,
            }
            by_ticker[t] = entry

        form = f.get("form", "")
        entry["counts"][form] = entry["counts"].get(form, 0) + 1
        entry["total"] += 1

        fd = f.get("filing_date") or ""
        if fd and (entry["latest_filing_date"] is None or fd > entry["latest_filing_date"]):
            entry["latest_filing_date"] = fd

    return by_ticker


def _filing_to_summary(f: Dict[str, Any]) -> FilingSummary:
    return FilingSummary(
        ticker=f.get("ticker", ""),
        company_name=f.get("company_name", ""),
        form=f.get("form", ""),
        fiscal_year_label=f.get("fiscal_year_label"),
        quarter_label=f.get("quarter_label"),
        period_label=f.get("period_label", ""),
        filing_date=f.get("filing_date", ""),
        accession=f.get("accession", ""),
        path=f.get("path", ""),
        section_count=f.get("section_count", 0),
        exhibit_count=f.get("exhibit_count", 0),
        filing_chars=f.get("filing_chars", 0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/status", response_class=ORJSONResponse, response_model=StatusResponse)
async def coverage_status() -> StatusResponse:
    """Lightweight summary for the 'Last refreshed' badge at the top of the Latest tab."""
    idx = _cache.get()
    stats = idx.get("stats", {})
    return StatusResponse(
        generated_at=idx.get("generated_at", ""),
        company_count=stats.get("company_count", 0),
        filing_count=stats.get("filing_count", 0),
        by_form=stats.get("by_form", {}),
    )


@router.get("/companies", response_class=ORJSONResponse, response_model=List[CompanySummary])
async def list_companies(
    q: Optional[str] = Query(None, description="Filter by ticker or company name (case-insensitive substring)"),
) -> List[CompanySummary]:
    """All 138 covered companies + their per-form filing counts and most recent filing date."""
    idx = _cache.get()
    by_ticker = _aggregate_company_stats(idx)
    rows = [CompanySummary(**v) for v in by_ticker.values()]

    # Sort: companies with filings first (by latest_filing_date desc), then alpha
    rows.sort(key=lambda r: (
        0 if r.latest_filing_date else 1,
        -ord(r.latest_filing_date[0]) if r.latest_filing_date else 0,  # rough recency tiebreak
        r.ticker,
    ))
    # Cleaner: actually sort by latest_filing_date desc, then ticker asc
    rows.sort(key=lambda r: (-(int((r.latest_filing_date or "0000-00-00").replace("-", ""))), r.ticker))

    if q:
        ql = q.strip().lower()
        if ql:
            rows = [r for r in rows if ql in r.ticker.lower() or ql in r.company_name.lower()]

    return rows


@router.get(
    "/companies/{ticker}",
    response_class=ORJSONResponse,
    response_model=CompanyDetail,
)
async def get_company(ticker: str) -> CompanyDetail:
    """Drill-down: every filing for one company, newest-first."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker required")

    idx = _cache.get()
    by_ticker = _aggregate_company_stats(idx)
    base = by_ticker.get(ticker)
    if base is None:
        raise HTTPException(status_code=404, detail=f"ticker {ticker!r} not in coverage")

    filings = [
        _filing_to_summary(f)
        for f in idx.get("filings", [])
        if f.get("ticker") == ticker
    ]
    # Index is already sorted newest-first overall; per-ticker order is preserved.

    return CompanyDetail(
        ticker=base["ticker"],
        cik=base["cik"],
        company_name=base["company_name"],
        counts=base["counts"],
        total=base["total"],
        filings=filings,
    )


class LatestFilingsResponse(BaseModel):
    items: List[FilingSummary]
    total: int        # post-filter total — for client-side pagination math
    offset: int
    limit: int


@router.get(
    "/latest",
    response_class=ORJSONResponse,
    response_model=LatestFilingsResponse,
)
async def latest_filings(
    limit: int = Query(50, ge=1, le=500, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="For pagination"),
    form: Optional[str] = Query(None, description="Filter to a single form: 10-K, 10-Q, or 8-K"),
    ticker: Optional[str] = Query(None, description="Filter to a single ticker"),
) -> LatestFilingsResponse:
    """Newest-first feed of filings across all coverage. Sorted by filing_date desc.

    Returns {items, total, offset, limit} so the frontend can render numbered
    pagination without doing a separate count query.
    """
    idx = _cache.get()
    items = idx.get("filings", [])

    if form:
        form = form.strip()
        items = [f for f in items if f.get("form") == form]
    if ticker:
        ticker = ticker.strip().upper()
        items = [f for f in items if f.get("ticker") == ticker]

    total = len(items)
    sliced = items[offset : offset + limit]
    return LatestFilingsResponse(
        items=[_filing_to_summary(f) for f in sliced],
        total=total,
        offset=offset,
        limit=limit,
    )
