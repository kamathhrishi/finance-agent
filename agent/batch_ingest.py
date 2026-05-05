#!/usr/bin/env python3
"""
Batch-ingest the tech universe into the agent corpus.

For each ticker × form combination, downloads the last `years` years of
filings (with substantive exhibits) and writes them into the corpus. Records
per-(ticker, form) status to a JSON checkpoint so re-runs skip work that
already completed and only retry the failures.

Usage:
    # Default — Tier A (Mega+Large IT/Comm), 5 years, all 3 forms, exhibits on
    python -m agent.batch_ingest

    # Subset of forms / different ticker scope / different year window
    python -m agent.batch_ingest --forms 10-K,10-Q --years 3
    python -m agent.batch_ingest --tickers MSFT,AAPL,GOOGL --years 5
    python -m agent.batch_ingest --no-exhibits

    # Retry just the ones that previously failed
    python -m agent.batch_ingest --retry-failed-only

Checkpoint:
    agent/data/_batch_checkpoint.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# IMPORTANT: SEC EDGAR throttles datamule's default user-agent for hot tickers
# (e.g. AAPL). Set a real identity before any datamule import.
os.environ.setdefault(
    "DATAMULE_SEC_USER_AGENT",
    "StrataLens kamathhrishi@gmail.com",
)

from agent.ingest import (    # noqa: E402
    SUPPORTED_FORMS,
    DATA_ROOT,
    ingest_form_for_ticker,
    write_data_readme,
    write_ticker_index,
    regenerate_index,
)
from agent.universe import load_universe  # noqa: E402

logger = logging.getLogger("agent.batch")

CHECKPOINT_PATH = DATA_ROOT / "_batch_checkpoint.json"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint state
# ─────────────────────────────────────────────────────────────────────────────


def _load_checkpoint() -> Dict[str, Dict[str, Any]]:
    if not CHECKPOINT_PATH.is_file():
        return {}
    try:
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_checkpoint(state: Dict[str, Dict[str, Any]]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _key(ticker: str, form: str) -> str:
    return f"{ticker}|{form}"


# ─────────────────────────────────────────────────────────────────────────────
# Batch driver
# ─────────────────────────────────────────────────────────────────────────────


def _ingest_one(
    ticker: str,
    form: str,
    years: int,
    keep_exhibits: bool,
) -> Tuple[bool, int, Optional[str]]:
    """Ingest one (ticker, form). Returns (ok, filings_written, error_msg)."""
    try:
        written = ingest_form_for_ticker(ticker, form, years, keep_exhibits=keep_exhibits)
        return True, len(written), None
    except Exception as e:
        return False, 0, f"{type(e).__name__}: {e}"


def run_batch(
    tickers: List[str],
    forms: List[str],
    years: int,
    keep_exhibits: bool,
    *,
    retry_failed_only: bool = False,
    skip_completed: bool = True,
) -> None:
    """Run the batch ingest with checkpointing."""
    state = _load_checkpoint()
    write_data_readme()

    n_total = len(tickers) * len(forms)
    n_processed = 0
    t_start = time.time()

    print(f"🟢 Batch ingest: {len(tickers)} tickers × {len(forms)} forms = {n_total} (ticker,form) jobs")
    print(f"   Years: last {years}, exhibits: {keep_exhibits}, retry_failed_only: {retry_failed_only}")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print()

    for i, ticker in enumerate(tickers, 1):
        for form in forms:
            n_processed += 1
            k = _key(ticker, form)
            prev = state.get(k)
            if prev:
                if retry_failed_only and prev.get("ok"):
                    continue
                if skip_completed and prev.get("ok") and not retry_failed_only:
                    print(f"  [{n_processed}/{n_total}] {ticker:<6} {form:<5}  ✓ skipping (already done, {prev.get('filings_written',0)} filings)")
                    continue

            t0 = time.time()
            print(f"  [{n_processed}/{n_total}] {ticker:<6} {form:<5}  ", end="", flush=True)
            ok, n_filings, err = _ingest_one(ticker, form, years, keep_exhibits)
            dt = time.time() - t0

            state[k] = {
                "ok": ok,
                "filings_written": n_filings,
                "error": err,
                "elapsed_s": round(dt, 1),
                "ts": int(time.time()),
                "years": years,
                "keep_exhibits": keep_exhibits,
            }
            _save_checkpoint(state)

            if ok:
                print(f"✅ {n_filings:>3} filings ({dt:.0f}s)")
            else:
                short = err[:120] if err else "?"
                print(f"❌ {short} ({dt:.0f}s)")

        # Refresh the per-ticker INDEX after every ticker so partial corpus is browsable
        try:
            write_ticker_index(ticker)
        except Exception as e:
            logger.warning(f"  per-ticker INDEX write failed for {ticker}: {e}")

    # Final top-level INDEX regen
    print("\n━━━ Regenerating top-level INDEX.md ━━━")
    regenerate_index()

    elapsed = time.time() - t_start
    ok_count = sum(1 for v in state.values() if v.get("ok"))
    fail_count = sum(1 for v in state.values() if not v.get("ok"))
    total_filings = sum(v.get("filings_written", 0) for v in state.values() if v.get("ok"))
    print(f"\n🏁 Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"   ok:    {ok_count}")
    print(f"   fail:  {fail_count}")
    print(f"   total filings written: {total_filings}")
    if fail_count:
        print(f"\n   Failures (re-run with --retry-failed-only to retry):")
        for k, v in sorted(state.items()):
            if not v.get("ok"):
                print(f"     {k:<14}  {v.get('error','?')[:100]}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, default=5, help="Year window (default: 5)")
    parser.add_argument("--forms", default=",".join(SUPPORTED_FORMS),
                        help=f"Comma-separated forms (default: {','.join(SUPPORTED_FORMS)})")
    parser.add_argument("--tickers", default="",
                        help="Override universe with explicit comma-separated tickers (e.g. MSFT,AAPL)")
    parser.add_argument("--no-exhibits", action="store_true", help="Skip exhibit downloading")
    parser.add_argument("--retry-failed-only", action="store_true",
                        help="Only retry (ticker,form) jobs that previously failed")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-ingest everything, ignoring checkpoint state")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.tickers.strip():
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = load_universe()

    forms = [f.strip() for f in args.forms.split(",") if f.strip()]
    bad = [f for f in forms if f not in SUPPORTED_FORMS]
    if bad:
        parser.error(f"Unsupported forms: {bad}. Supported: {SUPPORTED_FORMS}")

    run_batch(
        tickers=tickers,
        forms=forms,
        years=args.years,
        keep_exhibits=not args.no_exhibits,
        retry_failed_only=args.retry_failed_only,
        skip_completed=not args.no_skip,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
