"""
Load the FinanceBench dataset (HuggingFace `PatronusAI/financebench`) and
normalize each question into a record the runner can consume:

    FinanceBenchQuestion
        id            financebench_id (string, stable across runs)
        question      the user-facing question
        expected      the gold answer
        justification ground-truth explanation (used by judge for nuance)
        company       FinanceBench's company display name (e.g. "3M")
        ticker        SEC ticker we map company → (e.g. "MMM")
        form          "10-K" / "10-Q" / "8-K"  (uppercase, hyphen — matches our corpus)
        year          period year (int)
        quarter       Q1/Q2/Q3/Q4 if 10-Q else None
        filing_date   8-K's filing date (YYYY-MM-DD) if 8-K else None
        doc_name      original FinanceBench doc identifier (kept for traceability)

The transcript-only ("Earnings") questions are returned with `form = None`;
runners should filter them out unless transcripts are added to the corpus.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Company name → SEC ticker map. Hand-curated from the 32 companies that appear
# in FinanceBench. Keep these aligned to the strings FinanceBench actually emits
# in the `company` column (case-sensitive).
# ──────────────────────────────────────────────────────────────────────────────

# Excluded companies: Microsoft acquired Activision Blizzard (Oct 2023) →
# ATVI ticker is no longer in SEC's company_tickers.json. Block renamed from
# Square to a new ticker XYZ in 2024 → SQ no longer resolves. Both filings
# CAN be fetched by historical CIK, but until that adapter exists, we exclude
# them from the benchmark to avoid silent missing-data failures.
COMPANY_TO_TICKER: Dict[str, str] = {
    "3M": "MMM",
    "AES Corporation": "AES",
    "AMD": "AMD",
    # "Activision Blizzard": "ATVI",        # excluded: ticker delisted post-MSFT acquisition
    "Adobe": "ADBE",
    "Amazon": "AMZN",
    "Amcor": "AMCR",
    "American Express": "AXP",
    "American Water Works": "AWK",
    "Best Buy": "BBY",
    # "Block": "SQ",                        # excluded: SEC ticker renamed to XYZ in 2024
    "Boeing": "BA",
    "CVS Health": "CVS",
    "Coca-Cola": "KO",
    "Corning": "GLW",
    "Costco": "COST",
    "Foot Locker": "FL",
    "General Mills": "GIS",
    "JPMorgan": "JPM",
    "Johnson & Johnson": "JNJ",
    "Kraft Heinz": "KHC",
    "Lockheed Martin": "LMT",
    "MGM Resorts": "MGM",
    "Microsoft": "MSFT",
    "Netflix": "NFLX",
    "Nike": "NKE",
    "Paypal": "PYPL",
    "PepsiCo": "PEP",
    "Pfizer": "PFE",
    "Ulta Beauty": "ULTA",
    "Verizon": "VZ",
    "Walmart": "WMT",
}


@dataclass
class FinanceBenchQuestion:
    id: str
    question: str
    expected: str
    justification: str
    company: str
    ticker: Optional[str]
    form: Optional[str]            # "10-K" / "10-Q" / "8-K"  (None for transcripts)
    year: Optional[int]
    quarter: Optional[str]         # "Q1".."Q4" for 10-Q
    filing_date: Optional[str]     # YYYY-MM-DD for 8-K
    doc_name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Patterns FinanceBench uses for doc_name. Companies often have spaces removed
# or replaced with underscores, e.g. "JOHNSON_JOHNSON_2023_8K_dated-2023-08-30".
_DOC_8K_RE = re.compile(r"_(?P<year>\d{4})_8K_dated[-_](?P<date>\d{4}-\d{2}-\d{2})$", re.IGNORECASE)
_DOC_10Q_RE = re.compile(r"_(?P<year>\d{4})Q(?P<q>[1-4])_10Q$", re.IGNORECASE)
_DOC_10K_RE = re.compile(r"_(?P<year>\d{4})_10K$", re.IGNORECASE)


def _parse_doc(doc_name: str, doc_type_raw: str) -> Dict[str, Optional[str]]:
    """Pull (form, year, quarter, filing_date) from the FinanceBench doc_name."""
    out: Dict[str, Optional[str]] = {"form": None, "year": None, "quarter": None, "filing_date": None}
    dt = (doc_type_raw or "").lower()

    if dt == "10k" or _DOC_10K_RE.search(doc_name):
        out["form"] = "10-K"
        m = _DOC_10K_RE.search(doc_name)
        if m:
            out["year"] = int(m.group("year"))
        return out

    if dt == "10q" or _DOC_10Q_RE.search(doc_name):
        out["form"] = "10-Q"
        m = _DOC_10Q_RE.search(doc_name)
        if m:
            out["year"] = int(m.group("year"))
            out["quarter"] = f"Q{m.group('q')}"
        return out

    if dt == "8k" or _DOC_8K_RE.search(doc_name):
        out["form"] = "8-K"
        m = _DOC_8K_RE.search(doc_name)
        if m:
            out["year"] = int(m.group("year"))
            out["filing_date"] = m.group("date")
        return out

    # Earnings transcripts ("Earnings") — not currently in the corpus
    if dt == "earnings":
        return out  # form stays None

    return out


def load_questions(
    *,
    include_transcripts: bool = False,
    only_companies: Optional[List[str]] = None,
) -> List[FinanceBenchQuestion]:
    """Fetch and normalize FinanceBench. Returns the runnable subset by default."""
    from datasets import load_dataset

    raw = load_dataset("PatronusAI/financebench", split="train")
    out: List[FinanceBenchQuestion] = []
    for r in raw:
        company = (r.get("company") or "").strip()
        if only_companies and company not in only_companies:
            continue
        ticker = COMPANY_TO_TICKER.get(company)
        parsed = _parse_doc(r.get("doc_name") or "", r.get("doc_type") or "")
        form = parsed["form"]
        if not include_transcripts and form is None:
            continue
        # If we don't know the ticker (shouldn't happen with curated list), skip
        if ticker is None:
            continue
        out.append(FinanceBenchQuestion(
            id=str(r.get("financebench_id") or "").strip(),
            question=(r.get("question") or "").strip(),
            expected=(r.get("answer") or "").strip(),
            justification=(r.get("justification") or "").strip(),
            company=company,
            ticker=ticker,
            form=form,
            year=parsed["year"],
            quarter=parsed["quarter"],
            filing_date=parsed["filing_date"],
            doc_name=r.get("doc_name") or "",
        ))
    return out


def required_filings(questions: List[FinanceBenchQuestion]) -> List[Dict[str, Any]]:
    """
    Compute the deduped set of (ticker, form, year, quarter|filing_date) tuples
    needed to answer every question in `questions`. The download script ensures
    each of these exists in the corpus.
    """
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for q in questions:
        if not (q.ticker and q.form):
            continue
        key = (q.ticker, q.form, q.year, q.quarter, q.filing_date)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "ticker": q.ticker,
            "form": q.form,
            "year": q.year,
            "quarter": q.quarter,
            "filing_date": q.filing_date,
        })
    return out
