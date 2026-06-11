#!/usr/bin/env python3
"""
coverage_index — single source of truth for the Companies and Latest tabs.

Walks every `metadata.json` under data/filings/, joins with tech_universe.json,
and writes a flat JSON index to data/_coverage_index.json that the API router
loads into memory.

The index is rebuilt at three points:
  1. Manually via `python -m agent.coverage_index build`
  2. After every watcher cycle that wrote at least one new filing
  3. After every batch-ingest run

The JSON format (version 1):
  {
    "version": 1,
    "generated_at": "<ISO8601 UTC>",
    "stats": {
      "company_count": 138,
      "filing_count": 12206,
      "by_form": {"10-K": 552, "10-Q": 1656, "8-K": 9998}
    },
    "companies": [
      {"ticker": "AAPL", "cik": "...", "company_name": "Apple Inc."},
      ...
    ],
    "filings": [
      {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "form": "10-K",
        "fiscal_year_label": "FY2024",
        "quarter_label": null,
        "period_label": "FY2024",
        "filing_date": "2024-11-01",
        "accession": "0000320193-24-000123",
        "path": "filings/AAPL/10-K/FY2024",
        "filing_chars": 412345
      },
      ...
    ]
  }

`filing_date` is the date the SEC received the document — what users mean by
"when was this filed" — and is the default sort key for the Latest tab.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("agent.coverage_index")

PKG_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PKG_ROOT / "data"
DEFAULT_UNIVERSE_PATH = PKG_ROOT / "tech_universe.json"
INDEX_FILENAME = "_coverage_index.json"
INDEX_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _data_root_from_env() -> Path:
    env = os.getenv("FS_RESEARCH_DATA_ROOT", "").strip()
    return Path(env).resolve() if env else DEFAULT_DATA_ROOT


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_optional_text(value: Any) -> Optional[str]:
    text = _clean_text(value)
    return text or None


def _clean_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _period_label(form: str, fy: Optional[str], q: Optional[str], filing_date: str) -> str:
    """Render a human-readable period label that fits in a single table cell."""
    if form == "10-K" and fy:
        return fy                                  # "FY2025"
    if form == "10-Q" and fy and q:
        return f"{fy} {q}"                         # "FY2025 Q3"
    if form == "10-Q" and fy:
        return fy                                  # rare: 10-Q without quarter
    if form == "8-K":
        return f"filed {filing_date}" if filing_date else "8-K"
    return fy or filing_date or form


def _load_universe(universe_path: Path) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    """Return (companies_list, ticker→company_name map)."""
    if not universe_path.is_file():
        logger.warning(f"tech_universe.json missing at {universe_path}; companies list will be empty")
        return [], {}
    raw = json.loads(universe_path.read_text(encoding="utf-8"))
    tickers = raw.get("tickers") or []
    companies = []
    for t in tickers:
        ticker = _clean_text(t.get("ticker")).upper()
        if not ticker:
            continue
        companies.append({
            "ticker": ticker,
            "cik": _clean_text(t.get("cik")),
            "company_name": _clean_text(t.get("company_name")),
        })
    name_map = {c["ticker"]: c["company_name"] for c in companies}
    return companies, name_map


# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────


def build_index(
    data_root: Optional[Path] = None,
    universe_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Walk the corpus + return the full index dict (does not write it)."""
    data_root = (data_root or _data_root_from_env()).resolve()
    universe_path = (universe_path or DEFAULT_UNIVERSE_PATH).resolve()
    filings_root = data_root / "filings"

    companies, name_map = _load_universe(universe_path)
    filings: List[Dict[str, Any]] = []
    by_form: Dict[str, int] = {}

    if filings_root.is_dir():
        for meta_path in filings_root.rglob("metadata.json"):
            try:
                m = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"skipping unreadable metadata: {meta_path} ({e})")
                continue

            ticker = _clean_text(m.get("ticker")).upper()
            form = _clean_text(m.get("form")).upper()
            if not ticker or not form:
                continue

            fy = _clean_optional_text(m.get("fiscal_year_label"))
            q = _clean_optional_text(m.get("quarter_label"))
            filing_date = _clean_text(m.get("filing_date"))
            accession = _clean_text(m.get("accession"))
            chars = _clean_int(m.get("filing_chars"))
            section_keys = m.get("section_keys") or []
            exhibits = m.get("exhibits") or []

            # Path stored relative to data_root so the API can serve it without
            # leaking the absolute mount path.
            try:
                rel_dir = meta_path.parent.relative_to(data_root).as_posix()
            except ValueError:
                rel_dir = meta_path.parent.as_posix()

            filings.append({
                "ticker": ticker,
                "company_name": name_map.get(ticker, ""),
                "form": form,
                "fiscal_year_label": fy,
                "quarter_label": q,
                "period_label": _period_label(form, fy, q, filing_date),
                "filing_date": filing_date,
                "accession": accession,
                "path": rel_dir,
                "filing_chars": chars,
                "section_count": len(section_keys) if isinstance(section_keys, list) else 0,
                "exhibit_count": len(exhibits) if isinstance(exhibits, list) else 0,
            })
            by_form[form] = by_form.get(form, 0) + 1

    # Sort filings newest-first by filing_date (then ticker, form for stability)
    filings.sort(key=lambda f: (f["filing_date"] or "", f["ticker"], f["form"]), reverse=True)

    index = {
        "version": INDEX_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stats": {
            "company_count": len(companies),
            "filing_count": len(filings),
            "by_form": dict(sorted(by_form.items())),
        },
        "companies": companies,
        "filings": filings,
    }
    return index


def write_index(data_root: Optional[Path], index: Dict[str, Any]) -> Path:
    """Atomically write the index JSON to disk and return the path."""
    data_root = (data_root or _data_root_from_env()).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    out_path = data_root / INDEX_FILENAME
    # Atomic write: tmpfile + rename
    fd, tmp_name = tempfile.mkstemp(dir=str(data_root), prefix=".coverage_index.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        os.replace(tmp_name, out_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return out_path


def rebuild(data_root: Optional[Path] = None) -> Dict[str, Any]:
    """Convenience: build + write. Returns the index dict."""
    t0 = time.time()
    idx = build_index(data_root=data_root)
    out = write_index(data_root, idx)
    elapsed = time.time() - t0
    logger.info(
        f"coverage_index rebuilt: {idx['stats']['filing_count']} filings, "
        f"{idx['stats']['company_count']} companies, by_form={idx['stats']['by_form']}, "
        f"wrote {out} in {elapsed:.2f}s"
    )
    return idx


def load_index(data_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Read the on-disk index if present, else None."""
    data_root = (data_root or _data_root_from_env()).resolve()
    p = data_root / INDEX_FILENAME
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"coverage_index load failed: {e}")
        return None


def index_path(data_root: Optional[Path] = None) -> Path:
    """Where the index lives on disk (used by the API router for mtime checks)."""
    data_root = (data_root or _data_root_from_env()).resolve()
    return data_root / INDEX_FILENAME


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Coverage index builder for the Companies/Latest tabs.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="(Re)build the index and write _coverage_index.json")
    sub.add_parser("show", help="Print summary of the existing index")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "build":
        idx = rebuild()
        print(json.dumps(idx["stats"], indent=2))
        return 0

    if args.cmd == "show":
        idx = load_index()
        if not idx:
            print("(no index on disk yet — run `build`)", file=sys.stderr)
            return 1
        print(f"generated_at:  {idx.get('generated_at')}")
        print(f"version:       {idx.get('version')}")
        print(f"stats:         {json.dumps(idx.get('stats'), indent=2)}")
        print(f"companies:     {len(idx.get('companies', []))} entries")
        print(f"filings:       {len(idx.get('filings', []))} entries")
        first = (idx.get("filings") or [])[:3]
        print("first 3 filings (newest-first):")
        for f in first:
            print(f"  {f.get('filing_date'):>10}  {f.get('ticker'):<6}  {f.get('form'):<5}  {f.get('period_label'):<14}  {f.get('accession')}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(_cli())
