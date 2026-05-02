#!/usr/bin/env python3
"""
Relabel folders in an existing corpus to use the correct fiscal year /
quarter, derived from each filing's period_of_report (not the filing date).

Background: an earlier ingest bug labeled folders by `filing_date` because
period_of_report was empty in datamule's metadata. The fix to ingest reads
`meta["period"]` (YYYYMMDD) instead, but the existing 12K filings on disk
are still mislabeled (e.g. an MMM 10-K filed Feb 2018 covering FY2017 sits
under `FY2018/`).

This script:
  1. For each ticker in the corpus, fetches SEC's submissions JSON
     (https://data.sec.gov/submissions/CIK<padded>.json) — one request per
     ticker, contains accessionNumber + reportDate for every recent filing.
  2. For each existing filing folder, looks up the correct period via its
     accession number stored in metadata.json.
  3. Computes the correct FY (and Q for 10-Q) folder name.
  4. Renames the folder if it doesn't match. Idempotent.
  5. Updates metadata.json with the corrected fields.
  6. Regenerates the per-ticker INDEX and the top-level INDEX at the end.

Usage:
    # Dry-run first (recommended)
    python -m fs_research_agent.relabel_corpus --dry-run

    # Apply
    python -m fs_research_agent.relabel_corpus

    # Limit to specific tickers
    python -m fs_research_agent.relabel_corpus --tickers MSFT,AAPL,NVDA

    # Point at a different corpus root
    python -m fs_research_agent.relabel_corpus --data-root /path/to/data
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

# SEC requires a User-Agent identifying the requester
SEC_UA = os.getenv("DATAMULE_SEC_USER_AGENT", "StrataLens kamathhrishi@gmail.com")

from fs_research_agent.ingest import (    # noqa: E402
    DEFAULT_DATA_ROOT,
    SUPPORTED_FORMS,
    _normalize_date,
    _resolve_fiscal_year,
    _resolve_quarter,
    write_ticker_index,
    regenerate_index,
)

logger = logging.getLogger("fs_research_agent.relabel")


# ──────────────────────────────────────────────────────────────────────────────
# SEC submissions JSON
# ──────────────────────────────────────────────────────────────────────────────

_TICKER_TO_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


def _fetch_ticker_to_cik() -> Dict[str, str]:
    """Return {TICKER: '0000012345'} (10-digit zero-padded CIK)."""
    headers = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}
    r = httpx.get(_TICKER_TO_CIK_URL, headers=headers, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    out: Dict[str, str] = {}
    for entry in data.values():
        t = (entry.get("ticker") or "").upper()
        if t:
            out[t] = str(entry.get("cik_str", "")).zfill(10)
    return out


def _fetch_submissions_for_cik(cik: str, *, client: httpx.Client) -> Dict[str, Dict[str, str]]:
    """
    Fetch SEC submissions JSON for one CIK; return a map keyed by accession
    number → {form, filing_date, period_of_report}.

    SEC's submissions endpoint returns the most recent ~1,000 filings; older
    ones live in `files[].name` references that need separate fetches. For our
    5-10y windows that's almost always within the recent set.
    """
    url = _SUBMISSIONS_URL.format(cik=cik)
    r = client.get(url)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    data = r.json()
    recent = data.get("filings", {}).get("recent", {})
    accs = recent.get("accessionNumber", [])
    forms = recent.get("form", [])
    fds = recent.get("filingDate", [])
    rds = recent.get("reportDate", [])
    out: Dict[str, Dict[str, str]] = {}
    for i, acc in enumerate(accs):
        # Normalize accession (datamule may store with or without dashes)
        acc_norm = (acc or "").replace("-", "")
        out[acc_norm] = {
            "form": forms[i] if i < len(forms) else "",
            "filing_date": _normalize_date(fds[i] if i < len(fds) else ""),
            "period_of_report": _normalize_date(rds[i] if i < len(rds) else ""),
        }
    # Also handle older filings via files[].name (paginated archives)
    for f in data.get("filings", {}).get("files", []) or []:
        name = f.get("name")
        if not name:
            continue
        try:
            r2 = client.get(f"https://data.sec.gov/submissions/{name}")
            r2.raise_for_status()
            d2 = r2.json()
        except Exception as e:
            logger.warning(f"  failed to fetch older submissions {name}: {e}")
            continue
        accs2 = d2.get("accessionNumber", [])
        forms2 = d2.get("form", [])
        fds2 = d2.get("filingDate", [])
        rds2 = d2.get("reportDate", [])
        for i, acc in enumerate(accs2):
            acc_norm = (acc or "").replace("-", "")
            out[acc_norm] = {
                "form": forms2[i] if i < len(forms2) else "",
                "filing_date": _normalize_date(fds2[i] if i < len(fds2) else ""),
                "period_of_report": _normalize_date(rds2[i] if i < len(rds2) else ""),
            }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Per-filing relabel
# ──────────────────────────────────────────────────────────────────────────────


def _correct_folder_name(form: str, period: str, filing_date: str) -> Optional[Path]:
    """Compute the correct relative path stem (the filing's leaf folder name).

    Returns None if we can't determine it (e.g. period is missing).
    """
    if form == "10-K":
        if not period:
            return None
        return Path(_resolve_fiscal_year(period))                              # FY####
    if form == "10-Q":
        if not period:
            return None
        return Path(_resolve_fiscal_year(period)) / _resolve_quarter(period)   # FY####/Q#
    if form == "8-K":
        # 8-Ks are date-keyed by filing_date (event date). Period_of_report
        # is also relevant for 8-Ks but we use filing_date for the folder
        # name to maintain stable URLs.
        d = filing_date or period
        return Path(d) if d else None
    return None


def _relabel_one_filing(
    *,
    ticker: str,
    form: str,
    current_dir: Path,
    correct_meta: Dict[str, str],
    form_root: Path,
    dry_run: bool,
) -> Tuple[str, Optional[Path]]:
    """Relabel one filing's folder. Returns (status_str, new_dir_or_None)."""
    period = correct_meta.get("period_of_report", "")
    filing_date = correct_meta.get("filing_date", "")
    new_leaf = _correct_folder_name(form, period, filing_date)
    if new_leaf is None:
        return ("skip-no-period", None)

    new_dir = form_root / new_leaf
    if new_dir.resolve() == current_dir.resolve():
        # Already at the right place — just refresh metadata fields below
        status = "unchanged"
    else:
        if new_dir.exists():
            return ("conflict-target-exists", new_dir)
        if dry_run:
            return ("would-move", new_dir)
        new_dir.parent.mkdir(parents=True, exist_ok=True)
        current_dir.rename(new_dir)
        status = "moved"

    # Update metadata.json fields (period_of_report and filing_date) regardless
    if not dry_run:
        target_dir = new_dir if status != "unchanged" else current_dir
        meta_path = target_dir / "metadata.json"
        if meta_path.is_file():
            try:
                m = json.loads(meta_path.read_text(encoding="utf-8"))
                if period:
                    m["period_of_report"] = period
                if filing_date:
                    m["filing_date"] = filing_date
                if form == "10-K":
                    m["fiscal_year_label"] = _resolve_fiscal_year(period)
                elif form == "10-Q":
                    m["fiscal_year_label"] = _resolve_fiscal_year(period)
                    m["quarter_label"] = _resolve_quarter(period)
                meta_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
            except Exception as e:
                logger.warning(f"  failed to update {meta_path}: {e}")

    return (status, new_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────


def relabel_ticker(
    ticker: str,
    *,
    data_root: Path,
    submissions_map: Dict[str, Dict[str, str]],
    dry_run: bool,
) -> Dict[str, int]:
    """Relabel every filing folder for one ticker. Returns counters.

    Uses a two-pass move-via-staging approach so swapped destinations don't
    collide:
      Pass 1: every folder that needs to move → renamed to <form_root>/.stage-<accession>/
      Pass 2: from staging → final destination
    """
    counts = {"unchanged": 0, "moved": 0, "would-move": 0, "skip-no-period": 0,
              "skip-target-exists-and-different": 0, "no-metadata": 0, "no-accession-match": 0}

    ticker_root = data_root / "filings" / ticker
    if not ticker_root.is_dir():
        return counts

    for form in SUPPORTED_FORMS:
        form_root = ticker_root / form
        if not form_root.is_dir():
            continue

        # ── Plan: collect all (current_dir, target_dir, period, filing_date, acc) ──
        plan: List[Tuple[Path, Path, str, str, str]] = []
        for meta_path in sorted(form_root.rglob("metadata.json")):
            current_dir = meta_path.parent
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                counts["no-metadata"] += 1
                continue
            acc = (meta.get("accession") or "").replace("-", "")
            if not acc:
                counts["no-accession-match"] += 1
                continue
            correct = submissions_map.get(acc)
            if correct is None:
                counts["no-accession-match"] += 1
                continue
            period = correct.get("period_of_report", "")
            filing_date = correct.get("filing_date", "")
            new_leaf = _correct_folder_name(form, period, filing_date)
            if new_leaf is None:
                counts["skip-no-period"] += 1
                continue
            new_dir = form_root / new_leaf
            plan.append((current_dir, new_dir, period, filing_date, acc))

        # ── Execute plan with two-pass staging ──
        movers = [p for p in plan if p[0].resolve() != p[1].resolve()]
        unchanged = len(plan) - len(movers)
        counts["unchanged"] += unchanged
        if dry_run:
            counts["would-move"] += len(movers)
            continue
        if not movers:
            continue

        # Pass 1: rename to staging by accession (unique)
        stagings: List[Tuple[Path, Path, str, str]] = []   # (staging_dir, target_dir, period, filing_date, ...) re-stored
        plan_by_staging: Dict[Path, Tuple[Path, str, str]] = {}
        for current_dir, target_dir, period, filing_date, acc in movers:
            staging = form_root / f".stage-{acc}"
            try:
                if staging.exists():
                    # Stale staging from a previous failed run — move it out of the way
                    staging.rename(form_root / f".stale-{acc}-{int(time.time())}")
                current_dir.rename(staging)
                plan_by_staging[staging] = (target_dir, period, filing_date)
            except Exception as e:
                logger.warning(f"  pass-1 stage failed for {current_dir} → {staging}: {e}")

        # Pass 2: staging → final
        for staging, (target_dir, period, filing_date) in plan_by_staging.items():
            if target_dir.exists():
                # Should never happen if we staged everything first; defend anyway
                logger.warning(f"  pass-2 conflict: {target_dir} still exists; leaving {staging} alone")
                counts["skip-target-exists-and-different"] += 1
                continue
            try:
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                staging.rename(target_dir)
                # Update metadata.json with corrected fields
                meta_path = target_dir / "metadata.json"
                if meta_path.is_file():
                    try:
                        m = json.loads(meta_path.read_text(encoding="utf-8"))
                        if period:
                            m["period_of_report"] = period
                        if filing_date:
                            m["filing_date"] = filing_date
                        if form == "10-K":
                            m["fiscal_year_label"] = _resolve_fiscal_year(period)
                        elif form == "10-Q":
                            m["fiscal_year_label"] = _resolve_fiscal_year(period)
                            m["quarter_label"] = _resolve_quarter(period)
                        meta_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
                    except Exception as e:
                        logger.warning(f"  metadata refresh failed for {meta_path}: {e}")
                counts["moved"] += 1
            except Exception as e:
                logger.warning(f"  pass-2 move failed {staging} → {target_dir}: {e}")

        # Sweep: try to clean up empty parent dirs (e.g. FY2024 dir with no quarters left)
        for parent_dir in {p[1].parent for p in movers}:
            try:
                if parent_dir.is_dir() and not any(parent_dir.iterdir()):
                    parent_dir.rmdir()
            except OSError:
                pass

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be moved, but don't touch anything")
    parser.add_argument("--tickers", default="", help="Comma-separated ticker subset (default: all)")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Corpus root (default: main fs_research_agent corpus)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    data_root = Path(args.data_root).resolve()
    filings_root = data_root / "filings"
    if not filings_root.is_dir():
        print(f"❌ no filings dir at {filings_root}")
        return 1

    if args.tickers.strip():
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = sorted(p.name for p in filings_root.iterdir() if p.is_dir())

    print(f"📁 Corpus: {data_root}")
    print(f"🏷️  Tickers: {len(tickers)}")
    print(f"   Mode: {'DRY-RUN (no changes)' if args.dry_run else 'APPLY'}")
    print()

    print("Fetching SEC ticker → CIK map…")
    t2c = _fetch_ticker_to_cik()
    missing_tickers = [t for t in tickers if t not in t2c]
    if missing_tickers:
        print(f"  ⚠️ {len(missing_tickers)} ticker(s) not in SEC map: {missing_tickers[:8]}{' …' if len(missing_tickers) > 8 else ''}")

    aggregate: Dict[str, int] = {}
    headers = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}
    t0 = time.time()
    with httpx.Client(headers=headers, timeout=30.0) as client:
        for i, ticker in enumerate(tickers, 1):
            cik = t2c.get(ticker)
            if not cik:
                print(f"  [{i}/{len(tickers)}] {ticker:<6}  ⚠️ no CIK; skipping")
                continue
            try:
                subs = _fetch_submissions_for_cik(cik, client=client)
            except Exception as e:
                print(f"  [{i}/{len(tickers)}] {ticker:<6}  ❌ submissions fetch failed: {e}")
                continue
            time.sleep(0.12)  # SEC asks for <=10 req/s; we do <=8
            counts = relabel_ticker(ticker, data_root=data_root, submissions_map=subs, dry_run=args.dry_run)
            for k, v in counts.items():
                aggregate[k] = aggregate.get(k, 0) + v
            if not args.dry_run:
                try:
                    write_ticker_index(ticker, data_root=data_root)
                except Exception as e:
                    logger.warning(f"  per-ticker INDEX rewrite failed for {ticker}: {e}")
            interesting = {k: v for k, v in counts.items() if v and k != "unchanged"}
            tag = ", ".join(f"{k}={v}" for k, v in interesting.items()) if interesting else "all unchanged"
            print(f"  [{i}/{len(tickers)}] {ticker:<6}  {tag}")

    if not args.dry_run:
        print("\nRegenerating top-level INDEX.md…")
        regenerate_index(data_root=data_root)

    elapsed = time.time() - t0
    print(f"\n🏁 Done in {elapsed:.0f}s")
    print("Aggregate counts:")
    for k in sorted(aggregate):
        print(f"  {k:25}  {aggregate[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
