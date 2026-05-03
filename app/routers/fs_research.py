"""
FS Research router — serves the local markdown corpus consumed by
`fs_research_agent` and injects line-range highlight marks for the
SECFilingViewer to render.

Endpoints
─────────
GET  /fs-research/document?path=<rel>
    → {markdown, line_count, ticker, filing_type, fiscal_year, section}

POST /fs-research/document/with-highlights
    body: { path: str, relevant_chunks: [{chunk_id, line_start, line_end, primary?}] }
    → { document_markdown, highlighted_markdown, ... }   (shape mirrors /sec-filings/with-highlights)

The data root is `<repo>/fs_research_agent/data/` by default — override with
`FS_RESEARCH_DATA_ROOT` env var. All path inputs are resolved inside the
sandbox; anything escaping returns 400.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from fs_research_agent.highlight import LineHighlight, inject_line_highlights


router = APIRouter(prefix="/fs-research", tags=["fs-research"])


# Resolve the corpus root once at import time
def _resolve_data_root() -> Path:
    env = os.getenv("FS_RESEARCH_DATA_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
    else:
        # repo-relative default: <repo>/fs_research_agent/data/
        # __file__ is .../app/routers/fs_research.py → up 3 levels = repo root
        p = Path(__file__).resolve().parents[2] / "fs_research_agent" / "data"
    return p


_DATA_ROOT = _resolve_data_root()


def _safe_resolve(rel_path: str) -> Path:
    """Resolve a user-supplied path inside the sandbox. Raises on escape."""
    if not rel_path:
        raise HTTPException(status_code=400, detail="path is required")
    candidate = (_DATA_ROOT / rel_path).resolve()
    try:
        candidate.relative_to(_DATA_ROOT)
    except ValueError:
        raise HTTPException(status_code=400, detail="path escapes the corpus sandbox")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"file not found: {rel_path}")
    return candidate


# Matches all three corpus path shapes — must stay aligned with the
# canonical _PATH_RE in fs_research_agent/citations.py:
#   10-K:  filings/<TICKER>/10-K/FY####/(sections/|exhibits/)?<file>.md
#   10-Q:  filings/<TICKER>/10-Q/FY####/Q[1-4]/(sections/|exhibits/)?<file>.md
#   8-K:   filings/<TICKER>/8-K/YYYY-MM-DD/(sections/|exhibits/)?<file>.md
# The previous version only handled FY####/ — so 10-Q and 8-K paths
# returned an empty meta dict and the SECFilingViewer header rendered
# without ticker/filing_type/year (showed "Unknown FY 10-K").
_PATH_META_RE = re.compile(
    r"filings/(?P<ticker>[A-Z][A-Z0-9._-]{0,9})/(?P<form>10-K|10-Q|8-K)/"
    r"(?P<period>FY?\d{4}(?:/Q[1-4])?|\d{4}-\d{2}-\d{2})/"
    r"(?P<rest>(?:sections/|exhibits/)?[A-Za-z0-9._\-]+\.md)"
)


def _parse_meta(rel_path: str) -> dict:
    m = _PATH_META_RE.match(rel_path)
    if not m:
        return {}
    period = m.group("period")
    fy_digits = re.search(r"\d{4}", period or "")
    fy = int(fy_digits.group()) if fy_digits else None
    rest = m.group("rest")
    section = Path(rest).stem if rest.startswith("sections/") else None
    return {
        "ticker": m.group("ticker"),
        "filing_type": m.group("form"),
        "fiscal_year": fy,
        "section": section,
    }


# ─── Request models ──────────────────────────────────────────────────────────


class FsHighlightChunk(BaseModel):
    chunk_id: Optional[str] = None
    line_start: int = Field(..., ge=1)
    line_end: int = Field(..., ge=1)
    primary: bool = False


class FsHighlightRequest(BaseModel):
    path: str
    relevant_chunks: List[FsHighlightChunk] = Field(default_factory=list)


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.get("/document")
async def get_document(path: str = Query(..., description="Path relative to fs_research corpus root")):
    """Plain document fetch (no highlights)."""
    target = _safe_resolve(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    meta = _parse_meta(path)
    return ORJSONResponse({
        "success": True,
        "path": path,
        "document_markdown": text,
        "line_count": text.count("\n") + 1,
        **meta,
    })


@router.post("/document/with-highlights")
async def get_document_with_highlights(req: FsHighlightRequest):
    """Read a corpus file and wrap the cited line ranges in <mark>."""
    target = _safe_resolve(req.path)
    text = target.read_text(encoding="utf-8", errors="replace")

    highlights = [
        LineHighlight(
            chunk_id=c.chunk_id or f"FS-{i+1}",
            line_start=c.line_start,
            line_end=c.line_end,
            primary=c.primary,
        )
        for i, c in enumerate(req.relevant_chunks)
    ]
    highlighted = inject_line_highlights(text, highlights)

    meta = _parse_meta(req.path)
    return ORJSONResponse({
        "success": True,
        "path": req.path,
        "document_markdown": text,
        "highlighted_markdown": highlighted,
        "line_count": text.count("\n") + 1,
        # Shape parity with /sec-filings/with-highlights so SECFilingViewer
        # can consume both interchangeably:
        "company_name": meta.get("ticker"),
        "ticker": meta.get("ticker"),
        "filing_type": meta.get("filing_type"),
        "fiscal_year": meta.get("fiscal_year"),
        "section": meta.get("section"),
        "filing_date": None,
        "document_text": "",
        "document_length": len(text),
    })


@router.get("/health")
async def health():
    return {
        "ok": True,
        "data_root": str(_DATA_ROOT),
        "exists": _DATA_ROOT.is_dir(),
    }
