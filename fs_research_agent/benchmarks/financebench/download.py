"""
Build an ISOLATED FinanceBench corpus under
`fs_research_agent/benchmarks/financebench/data/`, mirroring the structure
of the main fs_research_agent corpus but containing ONLY the (ticker, form,
year/quarter/date) combinations FinanceBench questions reference.

Why isolation: makes the benchmark reproducible and prevents the agent from
accidentally answering from unrelated filings that happen to live in the
main tech corpus.

Layout (identical to main corpus):
    fs_research_agent/benchmarks/financebench/data/
      README.md
      INDEX.md
      filings/<TICKER>/<FORM>/...

Usage:
    from fs_research_agent.benchmarks.financebench.download import (
        ensure_required_filings, BENCHMARK_DATA_ROOT,
    )
    ensure_required_filings()                            # ALL questions
    ensure_required_filings(only_companies=["3M","AMD"]) # subset
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Force a real SEC user-agent before datamule import (otherwise SEC throttles AAPL etc.)
os.environ.setdefault("DATAMULE_SEC_USER_AGENT", "StrataLens kamathhrishi@gmail.com")

from fs_research_agent.ingest import (    # noqa: E402
    ingest_form_for_ticker,
    write_data_readme,
    write_ticker_index,
    regenerate_index,
)
from .dataset import (    # noqa: E402
    FinanceBenchQuestion,
    load_questions,
    required_filings,
)

logger = logging.getLogger("fs_research_agent.benchmarks.fb.download")

# The benchmark's isolated corpus root. Sibling of the agent code.
# Override with FS_RESEARCH_FINANCEBENCH_DATA_ROOT if you want it elsewhere.
BENCHMARK_DATA_ROOT: Path = Path(
    os.getenv(
        "FS_RESEARCH_FINANCEBENCH_DATA_ROOT",
        str(Path(__file__).resolve().parent / "data"),
    )
).resolve()


def _filings_root() -> Path:
    return BENCHMARK_DATA_ROOT / "filings"


def _have_filing(ticker: str, form: str, year: int, quarter: Optional[str], filing_date: Optional[str]) -> bool:
    """Best-effort check whether a specific FinanceBench-required filing is on disk."""
    base = _filings_root() / ticker / form
    if not base.is_dir():
        return False
    if form == "10-K":
        return (base / f"FY{year}").is_dir()
    if form == "10-Q":
        if not quarter:
            # Fallback: any quarter dir under FY<year>/
            return any((base / f"FY{year}").iterdir()) if (base / f"FY{year}").is_dir() else False
        return (base / f"FY{year}" / quarter).is_dir()
    if form == "8-K":
        if filing_date:
            return (base / filing_date).is_dir()
        # Fallback: any 8-K folder whose date starts with the year
        return any(p.name.startswith(str(year)) for p in base.iterdir())
    return False


def _years_window(years: List[int]) -> Tuple[int, int]:
    """Return (years_back_from_today, max_count) to cover the given year range."""
    if not years:
        return 5, 5
    now_year = datetime.utcnow().year
    span = max(now_year - min(years) + 2, 5)  # +2 safety buffer
    return span, span


def ensure_required_filings(
    *,
    questions: Optional[List[FinanceBenchQuestion]] = None,
    only_companies: Optional[List[str]] = None,
    keep_exhibits: bool = True,
    dry_run: bool = False,
) -> Dict[str, Dict[str, int]]:
    """
    Make sure every (ticker, form, year, quarter|date) referenced by the given
    questions exists in the corpus. Returns a per-(ticker, form) summary.
    """
    qs = questions if questions is not None else load_questions(only_companies=only_companies)
    needs = required_filings(qs)
    logger.info(f"FinanceBench needs {len(needs)} unique filings across {len(qs)} questions")

    # Group by (ticker, form). Year window per group spans all required years.
    grouped: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for n in needs:
        grouped[(n["ticker"], n["form"])].append(n)

    summary: Dict[str, Dict[str, int]] = defaultdict(lambda: {"needed": 0, "had": 0, "downloaded": 0, "still_missing": 0})

    BENCHMARK_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    write_data_readme(data_root=BENCHMARK_DATA_ROOT)
    logger.info(f"Benchmark corpus root: {BENCHMARK_DATA_ROOT}")

    for (ticker, form), items in sorted(grouped.items()):
        years = sorted({i["year"] for i in items if i.get("year")})
        span, _ = _years_window(years)
        per_ticker = summary[ticker]
        per_ticker["needed"] += len(items)

        # Quick on-disk check first — what do we already have?
        missing = [i for i in items if not _have_filing(ticker, form, i["year"], i.get("quarter"), i.get("filing_date"))]
        per_ticker["had"] += len(items) - len(missing)
        if not missing:
            logger.info(f"  ✓ {ticker} {form}: all {len(items)} filings already present")
            continue

        logger.info(f"  ↓ {ticker} {form}: {len(missing)}/{len(items)} missing — ingesting last {span}y into benchmark corpus")
        if dry_run:
            per_ticker["still_missing"] += len(missing)
            continue
        try:
            written = ingest_form_for_ticker(
                ticker, form, span,
                keep_exhibits=keep_exhibits,
                data_root=BENCHMARK_DATA_ROOT,
            )
            logger.info(f"     ↳ wrote {len(written)} {form} filing(s) for {ticker}")
        except Exception as e:
            logger.warning(f"     ↳ ingest failed for {ticker} {form}: {e}")

        # Re-check after download
        still_missing = [i for i in items if not _have_filing(ticker, form, i["year"], i.get("quarter"), i.get("filing_date"))]
        per_ticker["downloaded"] += len(missing) - len(still_missing)
        per_ticker["still_missing"] += len(still_missing)
        if still_missing:
            for sm in still_missing:
                logger.warning(f"     ↳ STILL missing: {sm}")

        write_ticker_index(ticker, data_root=BENCHMARK_DATA_ROOT)

    regenerate_index(data_root=BENCHMARK_DATA_ROOT)
    return dict(summary)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ensure FinanceBench filings are in the local corpus.")
    parser.add_argument("--companies", default="", help="Comma-separated subset of FinanceBench company names")
    parser.add_argument("--no-exhibits", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be downloaded but don't fetch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    only = [c.strip() for c in args.companies.split(",") if c.strip()] or None
    summary = ensure_required_filings(only_companies=only, keep_exhibits=not args.no_exhibits, dry_run=args.dry_run)
    print("\n━━━ SUMMARY ━━━")
    for ticker in sorted(summary):
        s = summary[ticker]
        print(f"  {ticker:6}  needed={s['needed']:>3}  had={s['had']:>3}  downloaded={s['downloaded']:>3}  still_missing={s['still_missing']:>3}")
