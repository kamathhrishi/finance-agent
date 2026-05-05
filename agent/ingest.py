#!/usr/bin/env python3
"""
Ingest SEC filings (10-K, 10-Q, 8-K + filtered exhibits) for a ticker into the
local research corpus.

Layout:
    data/
      README.md                                  ← orientation for the agent
      INDEX.md                                   ← top-level: ticker → counts
      filings/
        <TICKER>/
          INDEX.md                               ← per-ticker filing index
          10-K/<FY-LABEL>/                       ← e.g. FY2025
            filing.md
            metadata.json
            sections/<item>.md                   (when section parsing succeeds)
            exhibits/EX-<n>.md                   (filtered substantive exhibits)
          10-Q/<FY-LABEL>/<QUARTER>/             ← e.g. FY2025/Q3
            (same internal layout)
          8-K/<YYYY-MM-DD>/                      ← keyed by filing date
            (same internal layout)

Usage:
    python -m agent.ingest NVDA --years 5
    python -m agent.ingest NVDA --years 5 --forms 10-K,10-Q,8-K
    python -m agent.ingest NVDA --years 5 --no-exhibits
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("agent.ingest")

PKG_ROOT = Path(__file__).resolve().parent
# Default corpus root — the main agent corpus. Other corpora
# (e.g. the FinanceBench benchmark sandbox) pass an explicit `data_root` to
# all the public functions to write somewhere else.
DEFAULT_DATA_ROOT = PKG_ROOT / "data"
# Backwards-compat aliases — used by older callers and the API router.
DATA_ROOT = DEFAULT_DATA_ROOT
FILINGS_ROOT = DATA_ROOT / "filings"

SUPPORTED_FORMS = ("10-K", "10-Q", "8-K")


def _resolve_root(data_root: Optional[Path]) -> Path:
    """Resolve the corpus root for any ingest function. Default = main corpus."""
    if data_root is not None:
        return Path(data_root).resolve()
    return DEFAULT_DATA_ROOT


def _filings_root(data_root: Optional[Path] = None) -> Path:
    return _resolve_root(data_root) / "filings"


# ─────────────────────────────────────────────────────────────────────────────
# Section regex tables — one per form
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: (item_key, slug, regex). Regex matches a markdown heading
# (datamule emits H2/H3 like `## Item 1. Business`). Negative lookaheads on
# adjacent letters keep e.g. `Item 1` from matching `Item 1A`.

ITEMS_10K: List[Tuple[str, str, str]] = [
    ("1",   "item-1-business",                       r"^#{1,3}\s+Item\s+1\b(?!\s*[A-Z])"),
    ("1A",  "item-1a-risk-factors",                  r"^#{1,3}\s+Item\s+1A\b"),
    ("1B",  "item-1b-unresolved-staff-comments",     r"^#{1,3}\s+Item\s+1B\b"),
    ("1C",  "item-1c-cybersecurity",                 r"^#{1,3}\s+Item\s+1C\b"),
    ("2",   "item-2-properties",                     r"^#{1,3}\s+Item\s+2\b"),
    ("3",   "item-3-legal-proceedings",              r"^#{1,3}\s+Item\s+3\b"),
    ("4",   "item-4-mine-safety",                    r"^#{1,3}\s+Item\s+4\b"),
    ("5",   "item-5-market-for-registrants-equity",  r"^#{1,3}\s+Item\s+5\b"),
    ("6",   "item-6-selected-financial-data",        r"^#{1,3}\s+Item\s+6\b"),
    ("7",   "item-7-mda",                            r"^#{1,3}\s+Item\s+7\b(?!\s*A)"),
    ("7A",  "item-7a-quant-qual-disclosures",        r"^#{1,3}\s+Item\s+7A\b"),
    ("8",   "item-8-financial-statements",           r"^#{1,3}\s+Item\s+8\b"),
    ("9",   "item-9-changes-in-accountants",         r"^#{1,3}\s+Item\s+9\b(?!\s*[A-Z])"),
    ("9A",  "item-9a-controls-and-procedures",       r"^#{1,3}\s+Item\s+9A\b"),
    ("9B",  "item-9b-other-information",             r"^#{1,3}\s+Item\s+9B\b"),
    ("10",  "item-10-directors-and-officers",        r"^#{1,3}\s+Item\s+10\b"),
    ("11",  "item-11-executive-compensation",        r"^#{1,3}\s+Item\s+11\b"),
    ("12",  "item-12-security-ownership",            r"^#{1,3}\s+Item\s+12\b"),
    ("13",  "item-13-related-party-transactions",    r"^#{1,3}\s+Item\s+13\b"),
    ("14",  "item-14-principal-accountant-fees",     r"^#{1,3}\s+Item\s+14\b"),
    ("15",  "item-15-exhibits",                      r"^#{1,3}\s+Item\s+15\b"),
]

# 10-Q items appear in two parts. Item 1 in Part I (Financial Statements) is a
# different beast than Item 1 in Part II (Legal Proceedings). We bucket by Part.
ITEMS_10Q_PART1: List[Tuple[str, str, str]] = [
    ("p1-1",  "item-1-financial-statements",           r"^#{1,3}\s+Item\s+1\b(?!\s*[A-Z])"),
    ("p1-2",  "item-2-mda",                            r"^#{1,3}\s+Item\s+2\b"),
    ("p1-3",  "item-3-quant-qual-market-risk",         r"^#{1,3}\s+Item\s+3\b"),
    ("p1-4",  "item-4-controls-and-procedures",        r"^#{1,3}\s+Item\s+4\b"),
]
ITEMS_10Q_PART2: List[Tuple[str, str, str]] = [
    ("p2-1",  "item-1-legal-proceedings",              r"^#{1,3}\s+Item\s+1\b(?!\s*[A-Z])"),
    ("p2-1A", "item-1a-risk-factors",                  r"^#{1,3}\s+Item\s+1A\b"),
    ("p2-2",  "item-2-unregistered-equity-sales",      r"^#{1,3}\s+Item\s+2\b"),
    ("p2-3",  "item-3-defaults-on-senior-securities",  r"^#{1,3}\s+Item\s+3\b"),
    ("p2-4",  "item-4-mine-safety",                    r"^#{1,3}\s+Item\s+4\b"),
    ("p2-5",  "item-5-other-information",              r"^#{1,3}\s+Item\s+5\b"),
    ("p2-6",  "item-6-exhibits",                       r"^#{1,3}\s+Item\s+6\b"),
]

# 8-K items use decimal numbering (Item 2.02, 5.02, etc.). They're flat (no parts).
ITEMS_8K: List[Tuple[str, str, str]] = [
    ("1.01", "item-1-01-material-definitive-agreement",       r"^#{1,3}\s+Item\s+1\.01\b"),
    ("1.02", "item-1-02-termination-material-agreement",      r"^#{1,3}\s+Item\s+1\.02\b"),
    ("1.03", "item-1-03-bankruptcy",                          r"^#{1,3}\s+Item\s+1\.03\b"),
    ("2.01", "item-2-01-acquisition-disposition",             r"^#{1,3}\s+Item\s+2\.01\b"),
    ("2.02", "item-2-02-results-of-operations",               r"^#{1,3}\s+Item\s+2\.02\b"),
    ("2.03", "item-2-03-direct-financial-obligation",         r"^#{1,3}\s+Item\s+2\.03\b"),
    ("2.04", "item-2-04-triggering-events",                   r"^#{1,3}\s+Item\s+2\.04\b"),
    ("2.05", "item-2-05-exit-or-disposal",                    r"^#{1,3}\s+Item\s+2\.05\b"),
    ("2.06", "item-2-06-material-impairments",                r"^#{1,3}\s+Item\s+2\.06\b"),
    ("3.01", "item-3-01-delisting",                           r"^#{1,3}\s+Item\s+3\.01\b"),
    ("3.02", "item-3-02-unregistered-equity-sales",           r"^#{1,3}\s+Item\s+3\.02\b"),
    ("3.03", "item-3-03-rights-modification",                 r"^#{1,3}\s+Item\s+3\.03\b"),
    ("4.01", "item-4-01-accountant-change",                   r"^#{1,3}\s+Item\s+4\.01\b"),
    ("4.02", "item-4-02-non-reliance-on-financials",          r"^#{1,3}\s+Item\s+4\.02\b"),
    ("5.01", "item-5-01-change-in-control",                   r"^#{1,3}\s+Item\s+5\.01\b"),
    ("5.02", "item-5-02-officer-departure-election",          r"^#{1,3}\s+Item\s+5\.02\b"),
    ("5.03", "item-5-03-bylaw-amendments",                    r"^#{1,3}\s+Item\s+5\.03\b"),
    ("5.07", "item-5-07-shareholder-vote",                    r"^#{1,3}\s+Item\s+5\.07\b"),
    ("5.08", "item-5-08-shareholder-nominations",             r"^#{1,3}\s+Item\s+5\.08\b"),
    ("7.01", "item-7-01-regulation-fd",                       r"^#{1,3}\s+Item\s+7\.01\b"),
    ("8.01", "item-8-01-other-events",                        r"^#{1,3}\s+Item\s+8\.01\b"),
    ("9.01", "item-9-01-financial-statements-and-exhibits",   r"^#{1,3}\s+Item\s+9\.01\b"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Section parsing
# ─────────────────────────────────────────────────────────────────────────────


def _split_by_items(
    markdown: str,
    items: List[Tuple[str, str, str]],
    use_last_occurrence: bool = True,
    region: Optional[Tuple[int, int]] = None,
) -> Dict[str, str]:
    """
    Split markdown into sections using the given (key, slug, regex) table.

    `use_last_occurrence=True` picks the last match of each pattern (skips TOC
    entries that always come before the real section, used for 10-K).
    `use_last_occurrence=False` picks the first match (used inside a 10-Q part
    or for 8-K, where headings appear once each in the body).

    `region=(start, end)` restricts both the search and the section bodies to
    that slice of the markdown — used for 10-Q to confine items to a Part.
    """
    text = markdown[region[0]:region[1]] if region else markdown
    base_offset = region[0] if region else 0

    patterns = [(k, s, re.compile(rx, re.IGNORECASE | re.MULTILINE)) for (k, s, rx) in items]

    occurrences: List[Tuple[int, str, str]] = []
    for key, slug, pat in patterns:
        for m in pat.finditer(text):
            occurrences.append((m.start(), key, slug))

    if not occurrences:
        return {}

    occurrences.sort(key=lambda x: x[0])

    chosen_pos: Dict[str, Tuple[int, str]] = {}
    for pos, key, slug in occurrences:
        if use_last_occurrence:
            chosen_pos[key] = (pos, slug)        # overwrite to keep last
        else:
            chosen_pos.setdefault(key, (pos, slug))  # keep first

    timeline = sorted([(pos, key, slug) for key, (pos, slug) in chosen_pos.items()])

    sections: Dict[str, str] = {}
    for i, (pos, key, slug) in enumerate(timeline):
        end = timeline[i + 1][0] if i + 1 < len(timeline) else len(text)
        body = text[pos:end].strip()
        if len(body) >= 200:
            sections[key] = body
    return sections


def _split_10k(md: str) -> Dict[str, str]:
    """10-K sections — headings appear in TOC then again in body; pick last."""
    return _split_by_items(md, ITEMS_10K, use_last_occurrence=True)


def _split_10q(md: str) -> Dict[str, str]:
    """10-Q sections — split by Part I / Part II first, then items within each."""
    # Find Part headings. datamule emits like `## PART I` or `## PART II — OTHER…`.
    part_pat = re.compile(r"^#{1,3}\s+PART\s+(I{1,3})\b", re.IGNORECASE | re.MULTILINE)
    parts = list(part_pat.finditer(md))

    # Identify Part I and Part II positions. Last occurrence of each (skip TOC).
    p1_pos: Optional[int] = None
    p2_pos: Optional[int] = None
    for m in parts:
        roman = m.group(1).upper()
        if roman == "I":
            p1_pos = m.start()
        elif roman == "II":
            p2_pos = m.start()

    out: Dict[str, str] = {}
    if p1_pos is not None and p2_pos is not None and p2_pos > p1_pos:
        p1_region = (p1_pos, p2_pos)
        p2_region = (p2_pos, len(md))
        out.update(_split_by_items(md, ITEMS_10Q_PART1, use_last_occurrence=False, region=p1_region))
        out.update(_split_by_items(md, ITEMS_10Q_PART2, use_last_occurrence=False, region=p2_region))
    elif p1_pos is not None:
        # No Part II found — treat everything after Part I as Part I body
        out.update(_split_by_items(md, ITEMS_10Q_PART1, use_last_occurrence=False, region=(p1_pos, len(md))))
    else:
        # No Parts at all — try Part I patterns over the whole doc as fallback
        out.update(_split_by_items(md, ITEMS_10Q_PART1, use_last_occurrence=False))
    return out


def _split_8k(md: str) -> Dict[str, str]:
    """8-K sections — flat decimal items. Each appears once in body."""
    return _split_by_items(md, ITEMS_8K, use_last_occurrence=False)


def _slug_for(form: str, key: str) -> str:
    table = {"10-K": ITEMS_10K, "10-Q": ITEMS_10Q_PART1 + ITEMS_10Q_PART2, "8-K": ITEMS_8K}.get(form, [])
    for k, slug, _ in table:
        if k == key:
            return slug
    return f"item-{key.lower()}"


def _split_for_form(form: str, md: str) -> Dict[str, str]:
    if form == "10-K":
        return _split_10k(md)
    if form == "10-Q":
        return _split_10q(md)
    if form == "8-K":
        return _split_8k(md)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Exhibit filtering
# ─────────────────────────────────────────────────────────────────────────────

# Substantive exhibit type prefixes we keep. Everything else is dropped.
_EXHIBIT_KEEP_PREFIXES = ("EX-3.", "EX-10", "EX-19", "EX-21", "EX-99")
# Boilerplate / data-format exhibit type prefixes we explicitly skip
_EXHIBIT_SKIP_PREFIXES = (
    "EX-23",        # auditor consents (boilerplate)
    "EX-31",        # SOX certs (identical across companies)
    "EX-32",        # SOX certs (identical across companies)
    "EX-101",       # XBRL data
    "EX-104",       # cover page interactive data
)
# Non-text extensions we never bother with
_EXHIBIT_SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg",
    ".xlsx", ".xls", ".csv", ".pdf",
    ".zip", ".tar", ".gz",
    ".xsd", ".xml", ".css", ".js", ".json",
}


def _is_keepable_exhibit(doc_type: str, extension: str) -> bool:
    if not doc_type or not doc_type.upper().startswith("EX-"):
        return False
    t = doc_type.upper()
    if any(t.startswith(p) for p in _EXHIBIT_SKIP_PREFIXES):
        return False
    if any(t.startswith(p) for p in _EXHIBIT_KEEP_PREFIXES):
        ext = (extension or "").lower()
        if ext in _EXHIBIT_SKIP_EXTENSIONS:
            return False
        return True
    return False


def _safe_exhibit_filename(doc_type: str) -> str:
    """Turn EX-10.1 into a filesystem-safe slug like 'EX-10.1.md'."""
    base = re.sub(r"[^A-Za-z0-9._-]", "-", doc_type.strip())
    return f"{base}.md"


# ─────────────────────────────────────────────────────────────────────────────
# Path / fiscal-year / quarter resolution
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_date(date_str: str) -> str:
    """Coerce 'YYYYMMDD' or 'YYYY-MM-DD' (or 'YYYY-M-D') to canonical 'YYYY-MM-DD'.

    Returns '' if the input doesn't look like a date.
    """
    if not date_str:
        return ""
    s = date_str.strip()
    # Already canonical?
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    # YYYYMMDD?
    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return ""


def _resolve_fiscal_year(date_str: str) -> str:
    """Pick a fiscal-year label like 'FY2025' from any date-ish string."""
    norm = _normalize_date(date_str) or date_str
    m = re.match(r"(\d{4})", (norm or "").replace("-", ""))
    return f"FY{m.group(1)}" if m else "FY-unknown"


def _resolve_quarter(period_of_report: str) -> str:
    """
    Map a 10-Q period_of_report (YYYY-MM-DD or YYYYMMDD) to a calendar-quarter
    label Q1/Q2/Q3/Q4 based on the period-end month.

    NOTE: This is calendar quarter, not fiscal quarter. Companies with non-Dec
    fiscal years (e.g. NVDA, MSFT) have fiscal Q1/Q2/Q3 that don't align with
    calendar quarters. The README explains this so the agent doesn't assume
    Q3 = July-Sept for every ticker.
    """
    norm = _normalize_date(period_of_report)
    if not norm:
        return "Q-unknown"
    m = re.match(r"\d{4}-(\d{2})", norm)
    if not m:
        return "Q-unknown"
    month = int(m.group(1))
    if month <= 3: return "Q1"
    if month <= 6: return "Q2"
    if month <= 9: return "Q3"
    return "Q4"


def _filing_dir(
    form: str,
    ticker: str,
    filing_date: str,
    period_of_report: str,
    *,
    data_root: Optional[Path] = None,
) -> Path:
    """Resolve the on-disk directory for a single filing under the given corpus root."""
    base = _filings_root(data_root) / ticker / form
    fy = _resolve_fiscal_year(period_of_report or filing_date)
    if form == "10-K":
        return base / fy
    if form == "10-Q":
        return base / fy / _resolve_quarter(period_of_report or filing_date)
    if form == "8-K":
        # date-keyed (events, multiple per year); use filing_date for sortability
        date_key = filing_date or period_of_report or "unknown-date"
        return base / date_key
    return base / "misc"


# ─────────────────────────────────────────────────────────────────────────────
# Datamule download
# ─────────────────────────────────────────────────────────────────────────────


def _years_to_filing_date_range(years: int) -> Tuple[str, str]:
    """Return a (start, end) tuple for the last `years` years, ISO format."""
    end = datetime.utcnow()
    start = end - timedelta(days=years * 365 + 60)  # 60-day buffer for fiscal-year drift
    return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))


def _download_form_to_scratch(
    ticker: str,
    form: str,
    years: int,
    *,
    max_retries: int = 4,
) -> Path:
    """
    Download all `form` filings for `ticker` in the last `years` years.

    SEC EDGAR's EFTS search endpoint flakily returns 500s on rapid back-to-back
    queries (especially first request after idle). Retry with exponential
    backoff, since these are nearly always transient.
    """
    from datamule import Portfolio
    import time as _time
    import asyncio as _asyncio

    scratch = Path(tempfile.mkdtemp(prefix=f"datamule_{ticker}_{form.replace('-','')}_"))
    logger.info(f"  [{form}] scratch: {scratch}")

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            portfolio = Portfolio(str(scratch))
            portfolio.download_submissions(
                ticker=ticker,
                submission_type=form,
                filing_date=_years_to_filing_date_range(years),
                # NB: no document_type filter — we want exhibits too
            )
            return scratch
        except Exception as e:
            last_err = e
            msg = str(e)
            transient = (
                "500" in msg
                or "Internal Server Error" in msg
                or "ClientResponseError" in type(e).__name__
                or isinstance(e, _asyncio.TimeoutError)
            )
            if not transient or attempt == max_retries:
                raise
            wait = min(2 ** attempt, 30)  # 2, 4, 8, 16, then capped
            logger.warning(
                f"  [{form}] {ticker} attempt {attempt}/{max_retries} hit transient SEC error "
                f"({type(e).__name__}); retrying in {wait}s"
            )
            _time.sleep(wait)
    # Unreachable, but for type-checker
    raise last_err  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# Per-filing writer
# ─────────────────────────────────────────────────────────────────────────────


def _doc_markdown(doc) -> Optional[str]:
    """Best-effort extraction of markdown from a datamule document.

    Cleans table columns to collapse SEC's "$ + value" double-column pattern
    that confuses the LLM into reading the wrong fiscal year.
    """
    from .markdown_cleanup import cleanup_markdown_tables

    md: Optional[str] = None
    try:
        md = doc.markdown
    except Exception:
        md = None
    if not md or len(md) < 200:
        try:
            text = doc.text
            if text:
                md = str(text)
        except Exception:
            pass
    if not md:
        return None
    return cleanup_markdown_tables(md)


def _write_one_filing(
    sub,
    form: str,
    ticker: str,
    *,
    keep_exhibits: bool,
    data_root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Write one submission's main doc + exhibits to the corpus. Returns metadata or None on skip."""
    meta = sub.metadata.content if hasattr(sub.metadata, "content") else {}
    filing_date = _normalize_date(
        getattr(sub, "filing_date", None) or meta.get("filing-date", "") or ""
    )
    # datamule's submission metadata exposes the SEC `<PERIOD>` SGML field as
    # `meta["period"]` (a YYYYMMDD string — e.g. "20240928" for Apple's FY2024
    # 10-K). Older code looked at `period-of-report` / `period_of_report` which
    # are usually empty. Without this fallback every filing gets labeled by its
    # filing date instead of its fiscal year — i.e. a 10-K filed Feb 2018
    # covering FY2017 would be mislabeled as FY2018.
    period_of_report = _normalize_date(
        meta.get("period")
        or meta.get("period-of-report")
        or meta.get("period_of_report")
        or ""
    )
    cik = meta.get("cik") or meta.get("CIK") or ""
    accession = getattr(sub, "accession", "") or meta.get("accession-number", "")

    main_md: Optional[str] = None
    exhibits_to_write: List[Tuple[str, str, str]] = []  # (doc_type, extension, markdown)

    for d in sub:
        d_type = getattr(d, "type", "") or ""
        d_ext = getattr(d, "extension", "") or ""
        if d_type == form and main_md is None:
            md = _doc_markdown(d)
            if md:
                main_md = md
        elif keep_exhibits and _is_keepable_exhibit(d_type, d_ext):
            md = _doc_markdown(d)
            if md:
                exhibits_to_write.append((d_type, d_ext, md))

    if not main_md or len(main_md) < 1000:
        logger.warning(f"  [{form}] {ticker} {accession}: empty/short main doc; skipping")
        return None

    out_dir = _filing_dir(form, ticker, filing_date, period_of_report, data_root=data_root)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Main filing
    (out_dir / "filing.md").write_text(main_md, encoding="utf-8")

    # Sections
    sections = _split_for_form(form, main_md)
    if sections:
        (out_dir / "sections").mkdir(exist_ok=True)
        for key, body in sections.items():
            slug = _slug_for(form, key)
            (out_dir / "sections" / f"{slug}.md").write_text(body, encoding="utf-8")

    # Exhibits
    exhibits_meta: List[Dict[str, Any]] = []
    if exhibits_to_write:
        ex_dir = out_dir / "exhibits"
        ex_dir.mkdir(exist_ok=True)
        for ex_type, ex_ext, ex_md in exhibits_to_write:
            fname = _safe_exhibit_filename(ex_type)
            # Avoid clobbering when multiple exhibits share the same EX-N type
            target = ex_dir / fname
            n = 1
            while target.exists():
                n += 1
                target = ex_dir / f"{Path(fname).stem}__{n}.md"
            target.write_text(ex_md, encoding="utf-8")
            exhibits_meta.append({"type": ex_type, "file": target.name, "chars": len(ex_md)})

    meta_out = {
        "ticker": ticker,
        "cik": cik,
        "form": form,
        "fiscal_year_label": _resolve_fiscal_year(period_of_report or filing_date),
        # Use filing_date as fallback for quarter when period_of_report is missing
        # (matches what _filing_dir does — keeps the directory name and the
        # metadata.quarter_label field in sync).
        "quarter_label": _resolve_quarter(period_of_report or filing_date) if form == "10-Q" else None,
        "filing_date": filing_date,
        "period_of_report": period_of_report,
        "accession": accession,
        "filing_chars": len(main_md),
        "section_keys": sorted(sections.keys()) if sections else [],
        "exhibits": exhibits_meta,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    rel = out_dir.relative_to(_resolve_root(data_root))
    sec_str = f"{len(sections)} sections" if sections else "no sections (grep filing.md)"
    ex_str = f"{len(exhibits_meta)} exhibits" if exhibits_meta else "no exhibits"
    logger.info(f"  [{form}] ✓ {rel}  filed={filing_date}  {len(main_md):,} chars  {sec_str}, {ex_str}")
    return meta_out


# ─────────────────────────────────────────────────────────────────────────────
# Public ingest API
# ─────────────────────────────────────────────────────────────────────────────


def ingest_form_for_ticker(
    ticker: str,
    form: str,
    years: int,
    *,
    keep_exhibits: bool = True,
    max_filings: Optional[int] = None,
    data_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Download + write all `form` filings for `ticker` in the last `years` years.

    `data_root` selects the corpus to write into (default: the main fs_research
    corpus). Use this to populate an isolated benchmark sandbox.
    """
    from datamule import Portfolio

    if form not in SUPPORTED_FORMS:
        raise ValueError(f"Unsupported form {form!r}; supported: {SUPPORTED_FORMS}")

    ticker = ticker.upper()
    scratch = _download_form_to_scratch(ticker, form, years)

    portfolio = Portfolio(str(scratch))
    subs = list(portfolio)
    subs.sort(key=lambda s: getattr(s, "filing_date", "") or "", reverse=True)
    if max_filings:
        subs = subs[:max_filings]

    written: List[Dict[str, Any]] = []
    for sub in subs:
        try:
            m = _write_one_filing(sub, form, ticker, keep_exhibits=keep_exhibits, data_root=data_root)
            if m:
                written.append(m)
        except Exception as e:
            logger.warning(f"  [{form}] write failure for {ticker}: {type(e).__name__}: {e}")
            continue

    shutil.rmtree(scratch, ignore_errors=True)
    return written


def ingest_ticker(
    ticker: str,
    years: int,
    forms: Iterable[str] = SUPPORTED_FORMS,
    *,
    keep_exhibits: bool = True,
    data_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Download multiple form types for one ticker. Backwards-compatible API."""
    ticker = ticker.upper()
    all_written: List[Dict[str, Any]] = []
    per_form_counts: Dict[str, int] = {}
    errors: Dict[str, str] = {}

    for form in forms:
        try:
            written = ingest_form_for_ticker(
                ticker, form, years, keep_exhibits=keep_exhibits, data_root=data_root,
            )
            per_form_counts[form] = len(written)
            all_written.extend(written)
            logger.info(f"  [{form}] {ticker}: {len(written)} filing(s)")
        except Exception as e:
            errors[form] = f"{type(e).__name__}: {e}"
            per_form_counts[form] = 0
            logger.warning(f"  [{form}] {ticker} failed: {e}")

    if not all_written:
        raise RuntimeError(f"No filings written for {ticker} (errors: {errors})")

    write_ticker_index(ticker, data_root=data_root)
    return {
        "ticker": ticker,
        "filings": all_written,
        "per_form_counts": per_form_counts,
        "errors": errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# README + INDEX generation
# ─────────────────────────────────────────────────────────────────────────────


_README_BODY = """\
# Research corpus

SEC filings (10-K, 10-Q, 8-K) plus filtered material exhibits, organized for
filesystem-based research. The agent should read this file and `INDEX.md`
before grepping.

## Layout

```
filings/<TICKER>/
    INDEX.md                                   ← per-ticker index of every filing
    10-K/<FY-LABEL>/                           ← e.g. FY2025
        filing.md
        metadata.json
        sections/<item>.md                     ← when section parsing succeeds
        exhibits/EX-<n>.md                     ← filtered substantive exhibits
    10-Q/<FY-LABEL>/<QUARTER>/                 ← e.g. FY2025/Q3
        (same internal layout)
    8-K/<YYYY-MM-DD>/                          ← keyed by filing date (events)
        (same internal layout)
```

- `<TICKER>` is the company stock ticker (uppercase).
- `<FY-LABEL>` is `FY<YYYY>` derived from the filing's period of report.
- `<QUARTER>` is `Q1` / `Q2` / `Q3` / `Q4` from the **calendar** quarter of the
  period-end month. Note: companies with non-Dec fiscal years (e.g. NVDA's
  fiscal year ends in late January) will have calendar quarters that don't
  match their fiscal quarters. Always confirm by reading `metadata.json`.

## How to navigate efficiently

1. Read top-level `INDEX.md` to see what tickers are available and which forms
   each has.
2. Read `filings/<TICKER>/INDEX.md` to see every filing for one ticker. This
   file is small even for tickers with many filings.
3. Read a filing's `metadata.json` to confirm filing date, fiscal label, parsed
   section keys, and the list of exhibits.
4. Prefer reading specific sections or exhibits rather than the full
   `filing.md`. Use `grep` to localize before reading.
5. If `sections/` is empty (parsing fell back), use `grep` over `filing.md`.

## Form-specific tips

- **10-K** sections live in `sections/item-N-*.md`. MD&A is `item-7-mda.md`.
  Risk Factors is `item-1a-risk-factors.md`. Business overview is `item-1-business.md`.
- **10-Q** sections include Part I items (financial statements, MD&A) and
  Part II items (legal proceedings, risk factor updates). Slugs include the
  item number, e.g. `item-2-mda.md` (Part I MD&A) vs `item-1a-risk-factors.md`
  (Part II risk factors). Use the part context from the filename.
- **8-K** is event-driven: each filing covers specific items like 2.02
  (results of operations), 5.02 (officer departures), 7.01 (Reg FD), 9.01
  (financial statements & exhibits). Most 8-K substance is in `EX-99.1`
  exhibits (press releases). Always check `exhibits/` for 8-K filings.

## Exhibits — what's kept, what's not

Kept (substantive):
- `EX-3.x` — articles, bylaws (rare changes but material)
- `EX-10.x` — material contracts (credit, exec comp, supplier agreements)
- `EX-19` — insider trading policy (post-2024 SEC rule)
- `EX-21` — list of subsidiaries
- `EX-99.x` — press releases, financial supplements (highest-signal for 8-K)

Skipped (boilerplate or data-only):
- `EX-23` — auditor consents
- `EX-31` / `EX-32` — SOX certifications (boilerplate, identical across filers)
- `EX-101` / `EX-104` — XBRL / interactive data
- All non-text extensions (`.jpg`, `.xlsx`, `.zip`, etc.)

## Section parsing notes

Section parsing is regex-based on `Item N` markdown headings. When parsing
fails, `metadata.json` will show `"section_keys": []` — fall back to grep
over `filing.md`.
"""


def write_data_readme(data_root: Optional[Path] = None) -> None:
    root = _resolve_root(data_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text(_README_BODY, encoding="utf-8")


def _read_metadata_safe(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_ticker_index(ticker: str, data_root: Optional[Path] = None) -> None:
    """Write filings/<TICKER>/INDEX.md listing every filing for one ticker."""
    ticker = ticker.upper()
    root = _resolve_root(data_root)
    ticker_root = root / "filings" / ticker
    if not ticker_root.is_dir():
        return

    lines: List[str] = [f"# {ticker} — filing index", ""]
    forms = sorted(p.name for p in ticker_root.iterdir() if p.is_dir())
    for form in forms:
        form_root = ticker_root / form
        lines.append(f"## {form}")
        lines.append("")

        # Walk all filings under this form (depth varies: 10-K is 1, 10-Q is 2, 8-K is 1)
        meta_files: List[Path] = sorted(form_root.rglob("metadata.json"))
        if not meta_files:
            lines.append("_(no filings)_")
            lines.append("")
            continue
        # Sort newest first by filing_date in metadata
        def _key(mp: Path) -> str:
            return _read_metadata_safe(mp).get("filing_date", "") or ""
        meta_files.sort(key=_key, reverse=True)

        for mp in meta_files:
            m = _read_metadata_safe(mp)
            rel = mp.parent.relative_to(root)
            filed = m.get("filing_date") or "?"
            fy = m.get("fiscal_year_label") or ""
            q = m.get("quarter_label") or ""
            label = " ".join(x for x in [fy, q] if x).strip() or rel.name
            sect = m.get("section_keys") or []
            ex = m.get("exhibits") or []
            sect_str = f"{len(sect)} sections" if sect else "no sections"
            ex_str = f"{len(ex)} exhibits" if ex else "no exhibits"
            lines.append(f"- **{label}** (filed {filed}) — `{rel}/`")
            lines.append(f"    - {sect_str}, {ex_str}")
        lines.append("")

    (ticker_root / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def _company_name_from_metadata(ticker_root: Path) -> str:
    """Best-effort: pull the company display name from any metadata.json under
    this ticker. We look for `company_name` or `filer_name` first, then fall
    back to scanning a sample filing.md for the cover-page company name."""
    for mp in ticker_root.rglob("metadata.json"):
        m = _read_metadata_safe(mp)
        for key in ("company_name", "filer_name", "company", "name"):
            v = m.get(key)
            if v and isinstance(v, str) and len(v) > 0:
                return v.strip()
    return ""


# Hardcoded ticker → company display name map. Fallback when metadata doesn't
# carry the name. Covers the FinanceBench 32 + the larger tech universe.
# Used by `regenerate_index` to enrich the top-level INDEX.md so the agent
# can connect a question's company-name reference (e.g. "3M") to the corpus's
# ticker (e.g. "MMM") without guessing.
_TICKER_TO_COMPANY: Dict[str, str] = {
    "MMM": "3M Company", "AAPL": "Apple Inc.", "ADBE": "Adobe Inc.", "AES": "AES Corporation",
    "AMCR": "Amcor plc", "AMD": "Advanced Micro Devices", "AMZN": "Amazon.com",
    "ATVI": "Activision Blizzard", "AWK": "American Water Works", "AXP": "American Express",
    "BA": "Boeing", "BBY": "Best Buy", "COST": "Costco Wholesale",
    "CVS": "CVS Health", "DDOG": "Datadog", "FL": "Foot Locker", "GIS": "General Mills",
    "GLW": "Corning", "GOOGL": "Alphabet (Google)", "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase", "KHC": "Kraft Heinz", "KO": "Coca-Cola", "LMT": "Lockheed Martin",
    "META": "Meta Platforms", "MGM": "MGM Resorts", "MSFT": "Microsoft", "NFLX": "Netflix",
    "NKE": "Nike", "NVDA": "NVIDIA", "ORCL": "Oracle", "PEP": "PepsiCo", "PFE": "Pfizer",
    "PLTR": "Palantir Technologies", "PYPL": "PayPal", "SQ": "Block (formerly Square)",
    "ULTA": "Ulta Beauty", "VZ": "Verizon", "WMT": "Walmart",
    # tech-batch additions worth labeling
    "AVGO": "Broadcom", "CRM": "Salesforce", "INTC": "Intel", "QCOM": "Qualcomm",
    "CSCO": "Cisco", "NOW": "ServiceNow", "UBER": "Uber", "ABNB": "Airbnb",
    "COIN": "Coinbase", "SNOW": "Snowflake", "CRWD": "CrowdStrike", "PANW": "Palo Alto Networks",
    "MU": "Micron Technology", "AMAT": "Applied Materials", "LRCX": "Lam Research",
    "KLAC": "KLA Corporation", "MRVL": "Marvell Technology", "TSLA": "Tesla",
}


def regenerate_index(data_root: Optional[Path] = None) -> None:
    """Top-level INDEX.md: ticker + company name → form → count summary.

    Including the company name is critical: questions often reference a
    company by name (e.g. "3M", "Best Buy") while the corpus uses tickers
    (e.g. "MMM", "BBY"). Without the name in INDEX.md the agent can read MMM
    in the table and still conclude "3M is not in the corpus".
    """
    root = _resolve_root(data_root)
    filings_root = root / "filings"
    root.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Corpus index",
        "",
        "Top-level summary of every ticker in the corpus. For per-filing detail, read `filings/<TICKER>/INDEX.md`.",
        "",
        "**Note on company names vs tickers:** the corpus is keyed by SEC ticker. Questions that name a company (e.g. \"3M\", \"Best Buy\", \"Johnson & Johnson\") refer to the ticker shown in this table (e.g. MMM, BBY, JNJ). If the company you want appears in the **Company** column below, the data is here.",
        "",
        "| Ticker | Company | 10-K | 10-Q | 8-K | Latest filing | Per-ticker index |",
        "|--------|---------|-----:|-----:|----:|---------------|------------------|",
    ]
    if not filings_root.exists():
        lines.append("_(corpus is empty)_")
        (root / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
        return

    tickers = sorted(p.name for p in filings_root.iterdir() if p.is_dir())
    for ticker in tickers:
        ticker_root = filings_root / ticker
        # Count metadata.json files per form
        counts: Dict[str, int] = {f: 0 for f in SUPPORTED_FORMS}
        latest = ""
        for form in SUPPORTED_FORMS:
            form_root = ticker_root / form
            if not form_root.is_dir():
                continue
            for mp in form_root.rglob("metadata.json"):
                counts[form] += 1
                m = _read_metadata_safe(mp)
                fd = m.get("filing_date") or ""
                if fd > latest:
                    latest = fd
        # Resolve company name: prefer metadata, fall back to map, last resort = ticker
        company = _company_name_from_metadata(ticker_root) or _TICKER_TO_COMPANY.get(ticker, ticker)
        # Ensure per-ticker INDEX exists
        if not (ticker_root / "INDEX.md").exists():
            write_ticker_index(ticker, data_root=root)
        index_rel = (ticker_root / "INDEX.md").relative_to(root)
        lines.append(
            f"| {ticker} | {company} | {counts['10-K']} | {counts['10-Q']} | {counts['8-K']} | {latest or '—'} | `{index_rel}` |"
        )
    lines.append("")
    (root / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest SEC filings (10-K, 10-Q, 8-K + exhibits) for a ticker."
    )
    parser.add_argument("ticker", help="Ticker symbol, e.g. NVDA")
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Year window from today (default: 5).",
    )
    parser.add_argument(
        "--forms",
        default=",".join(SUPPORTED_FORMS),
        help=f"Comma-separated subset of forms to ingest. Default: {','.join(SUPPORTED_FORMS)}",
    )
    parser.add_argument("--no-exhibits", action="store_true", help="Skip exhibit downloading.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    forms = [f.strip() for f in args.forms.split(",") if f.strip()]
    write_data_readme()
    result = ingest_ticker(args.ticker, args.years, forms=forms, keep_exhibits=not args.no_exhibits)
    regenerate_index()

    print(f"\n✅ {result['ticker']}: ingested {len(result['filings'])} filing(s)")
    for form, n in result["per_form_counts"].items():
        print(f"   {form:6} {n} filing(s)")
    if result["errors"]:
        for form, err in result["errors"].items():
            print(f"   {form:6} ERROR: {err}")
    print(f"\n📁 Corpus: {DATA_ROOT}")
    print(f"📄 Index:  {DATA_ROOT / 'INDEX.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
