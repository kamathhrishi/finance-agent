#!/usr/bin/env python3
"""
Background watcher that polls SEC EDGAR for new filings across the canonical
tech universe and downloads anything new into the local corpus.

Loop per cycle:
  1. For each ticker in tech_universe.json:
     a. Fetch SEC's submissions JSON (data.sec.gov/submissions/CIK<padded>.json).
     b. List recent (form, accessionNumber, reportDate) tuples for forms
        we care about (default: 10-K, 10-Q, 8-K).
     c. Compare against our local "seen" set (state/_seen_accessions.json).
     d. For any new accession, download via datamule with `accession_numbers=[…]`
        — pulls just the one filing, writes into the corpus per the existing
        ingest pipeline (with column dedupe + exhibit filter applied).
     e. Refresh per-ticker INDEX.md.
  2. After the cycle finishes, regenerate the top-level INDEX.md if anything
     was downloaded.
  3. Sleep `--interval` seconds, repeat.

State files (created lazily under `data/`):
  _seen_accessions.json  — { "TICKER": ["accession1", "accession2", ...], ... }
  _watcher_state.json    — last-cycle stats + timestamps

Usage:
  python -m fs_research_agent.watcher                     # poll every 10 min
  python -m fs_research_agent.watcher --interval 1800     # every 30 min
  python -m fs_research_agent.watcher --forms 10-K,8-K    # subset of forms
  python -m fs_research_agent.watcher --once              # one cycle and exit

Run as a background daemon:
  nohup python -m fs_research_agent.watcher >/var/log/fs_watcher.log 2>&1 &

Or as a Railway cron job ("every 30 min"):
  python -m fs_research_agent.watcher --once
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import httpx

# Set SEC user-agent BEFORE importing datamule (it captures it at import time)
os.environ.setdefault(
    "DATAMULE_SEC_USER_AGENT",
    "StrataLens kamathhrishi@gmail.com",
)

from fs_research_agent.ingest import (    # noqa: E402
    DEFAULT_DATA_ROOT,
    SUPPORTED_FORMS,
    _normalize_date,
    _write_one_filing,
    write_data_readme,
    write_ticker_index,
    regenerate_index,
)
from fs_research_agent.tech_universe import TickerSpec, load_tech_universe  # noqa: E402

logger = logging.getLogger("fs_research_agent.watcher")

DEFAULT_POLL_INTERVAL_SECS = 600   # 10 minutes
DEFAULT_MAX_AGE_DAYS = 30          # only fetch filings filed within last N days
SEC_RATE_LIMIT_GAP_SECS = 0.15     # ~7 req/s (SEC asks for ≤10)
DEFAULT_FORMS = ("10-K", "10-Q", "8-K")
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


# ──────────────────────────────────────────────────────────────────────────────
# State persistence (minimal — just what's needed for diffing)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class CycleStats:
    cycle_started_at: str
    cycle_finished_at: str
    duration_s: float
    tickers_polled: int
    tickers_failed: int
    new_filings_found: int
    new_filings_downloaded: int
    download_failures: int


def _state_paths(data_root: Path) -> Tuple[Path, Path]:
    return data_root / "_seen_accessions.json", data_root / "_watcher_state.json"


def _load_seen(data_root: Path) -> Dict[str, Set[str]]:
    seen_path, _ = _state_paths(data_root)
    if not seen_path.is_file():
        return {}
    try:
        raw = json.loads(seen_path.read_text(encoding="utf-8"))
        return {k: set(v) for k, v in raw.items()}
    except Exception as e:
        logger.warning(f"failed to load seen-state at {seen_path}: {e} — starting fresh")
        return {}


def _save_seen(data_root: Path, seen: Dict[str, Set[str]]) -> None:
    seen_path, _ = _state_paths(data_root)
    seen_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: sorted(v) for k, v in seen.items()}
    seen_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _save_cycle_stats(data_root: Path, stats: CycleStats) -> None:
    _, state_path = _state_paths(data_root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(stats.__dict__, indent=2), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap "seen" from what's already on disk, so first run doesn't re-download
# 12,000 filings just to discover they're already there.
# ──────────────────────────────────────────────────────────────────────────────


def _bootstrap_seen_from_disk(
    data_root: Path,
    universe: List[TickerSpec],
) -> Dict[str, Set[str]]:
    """Walk the corpus and collect every accession number already on disk."""
    seen: Dict[str, Set[str]] = defaultdict(set)
    for spec in universe:
        ticker_root = data_root / "filings" / spec.ticker
        if not ticker_root.is_dir():
            continue
        for meta_path in ticker_root.rglob("metadata.json"):
            try:
                m = json.loads(meta_path.read_text(encoding="utf-8"))
                acc = (m.get("accession") or "").replace("-", "")
                if acc:
                    seen[spec.ticker].add(acc)
            except Exception:
                continue
    return seen


# ──────────────────────────────────────────────────────────────────────────────
# Per-ticker poll + diff
# ──────────────────────────────────────────────────────────────────────────────


async def _fetch_submissions(client: httpx.AsyncClient, cik: str) -> Optional[dict]:
    """Async fetch of the per-CIK submissions JSON. Returns None on 404."""
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    try:
        r = await client.get(url)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"submissions fetch failed for CIK {cik}: {e}")
        return None


def _list_recent_filings(
    submissions: dict,
    forms: Iterable[str],
    *,
    max_age_days: Optional[int] = None,
) -> List[dict]:
    """Pull (form, accession, filing_date, period) tuples from submissions JSON.

    If `max_age_days` is set, drops anything filed more than N days ago — keeps
    the watcher focused on recently-filed forms instead of retroactively
    backfilling older filings (use the `batch_ingest` tool for that).
    """
    from datetime import date, timedelta

    forms_set = {f.upper() for f in forms}
    cutoff = (date.today() - timedelta(days=max_age_days)).isoformat() if max_age_days else None

    recent = submissions.get("filings", {}).get("recent", {})
    accs = recent.get("accessionNumber", [])
    formvs = recent.get("form", [])
    fds = recent.get("filingDate", [])
    rds = recent.get("reportDate", [])
    out = []
    for i, acc in enumerate(accs):
        form = (formvs[i] if i < len(formvs) else "").upper()
        if form not in forms_set:
            continue
        fd = _normalize_date(fds[i] if i < len(fds) else "")
        if cutoff and fd and fd < cutoff:
            continue   # too old — leave it for an explicit backfill
        out.append({
            "accession_norm": (acc or "").replace("-", ""),
            "accession": acc,
            "form": form,
            "filing_date": fd,
            "period_of_report": _normalize_date(rds[i] if i < len(rds) else ""),
        })
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Download a single accession via datamule
# ──────────────────────────────────────────────────────────────────────────────


def _ingest_single_accession(
    *,
    ticker: str,
    cik: str,          # 10-digit zero-padded
    accession: str,    # WITH dashes (datamule's expected form)
    form: str,
    data_root: Path,
    keep_exhibits: bool,
) -> bool:
    """Download one specific filing (by accession) and write it into the corpus.

    Implementation note: datamule's `download_submissions(ticker=..., accession_numbers=[...])`
    IGNORES the accession_numbers filter and downloads every filing of that
    ticker+form. We work around it by passing `cik=...` only (no ticker, no
    submission_type) — that path correctly honors `accession_numbers`.
    Then we filter the resulting portfolio to the exact accession we asked for.
    """
    import shutil
    import tempfile
    from datamule import Portfolio

    scratch = Path(tempfile.mkdtemp(prefix=f"watcher_{ticker}_"))
    acc_norm = accession.replace("-", "")
    try:
        portfolio = Portfolio(str(scratch))
        portfolio.download_submissions(
            cik=cik,
            accession_numbers=[accession],
        )
        # Pick out the submission whose accession matches what we asked for
        match = None
        for sub in portfolio:
            sub_acc = (getattr(sub, "accession", "") or "").replace("-", "")
            if sub_acc == acc_norm:
                match = sub
                break
        if match is None:
            logger.warning(f"  {ticker} {form} {accession}: datamule returned no matching submission")
            return False
        meta = _write_one_filing(match, form, ticker, keep_exhibits=keep_exhibits, data_root=data_root)
        return meta is not None
    except Exception as e:
        logger.warning(f"  {ticker} {form} {accession}: ingest failed: {type(e).__name__}: {e}")
        return False
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────────
# Cycle driver
# ──────────────────────────────────────────────────────────────────────────────


async def run_one_cycle(
    *,
    universe: List[TickerSpec],
    data_root: Path,
    forms: Iterable[str],
    seen: Dict[str, Set[str]],
    keep_exhibits: bool,
    max_age_days: Optional[int] = None,
) -> CycleStats:
    started_at = datetime.now(timezone.utc)
    t0 = time.time()
    tickers_failed = 0
    new_total = 0
    downloaded = 0
    failed_downloads = 0
    user_agent = os.environ["DATAMULE_SEC_USER_AGENT"]
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    touched_tickers: Set[str] = set()

    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        for spec in universe:
            submissions = await _fetch_submissions(client, spec.cik)
            await asyncio.sleep(SEC_RATE_LIMIT_GAP_SECS)
            if submissions is None:
                tickers_failed += 1
                continue
            recent = _list_recent_filings(submissions, forms, max_age_days=max_age_days)
            ticker_seen = seen.setdefault(spec.ticker, set())
            new_for_this_ticker = [r for r in recent if r["accession_norm"] not in ticker_seen]
            if not new_for_this_ticker:
                continue
            new_total += len(new_for_this_ticker)
            logger.info(f"  {spec.ticker} ({spec.company_name}): {len(new_for_this_ticker)} new filing(s)")
            for f in new_for_this_ticker:
                ok = _ingest_single_accession(
                    ticker=spec.ticker,
                    cik=spec.cik,
                    accession=f["accession"],
                    form=f["form"],
                    data_root=data_root,
                    keep_exhibits=keep_exhibits,
                )
                if ok:
                    downloaded += 1
                    ticker_seen.add(f["accession_norm"])
                    touched_tickers.add(spec.ticker)
                    logger.info(f"     ✓ {f['form']} accession={f['accession']} period={f['period_of_report']}")
                else:
                    failed_downloads += 1

    # Refresh per-ticker indexes for everyone we touched
    if touched_tickers:
        write_data_readme(data_root=data_root)
        for t in sorted(touched_tickers):
            try:
                write_ticker_index(t, data_root=data_root)
            except Exception as e:
                logger.warning(f"  index refresh failed for {t}: {e}")
        try:
            regenerate_index(data_root=data_root)
        except Exception as e:
            logger.warning(f"  top-level INDEX refresh failed: {e}")
        _save_seen(data_root, seen)

        # Rebuild the coverage_index.json that powers the Companies / Latest
        # tabs so new filings show up in the UI on the next request without a
        # restart. Done last so partial state from earlier in the cycle isn't
        # exposed.
        try:
            from .coverage_index import rebuild as _rebuild_coverage
            _rebuild_coverage(data_root=data_root)
        except Exception as e:
            logger.warning(f"  coverage_index rebuild failed: {e}")

    finished_at = datetime.now(timezone.utc)
    return CycleStats(
        cycle_started_at=started_at.isoformat(timespec="seconds"),
        cycle_finished_at=finished_at.isoformat(timespec="seconds"),
        duration_s=round(time.time() - t0, 1),
        tickers_polled=len(universe),
        tickers_failed=tickers_failed,
        new_filings_found=new_total,
        new_filings_downloaded=downloaded,
        download_failures=failed_downloads,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────


async def watcher_loop(
    *,
    interval_secs: int,
    data_root: Path,
    forms: Iterable[str],
    keep_exhibits: bool,
    once: bool,
    max_age_days: Optional[int] = None,
    install_signal_handlers: bool = True,
) -> int:
    """Long-running poll loop. Set install_signal_handlers=False when running
    inside another process (e.g. as an asyncio.task spawned from FastAPI's
    lifespan) — otherwise we clobber uvicorn's SIGTERM/SIGINT handlers.
    Cancellation in that case happens via asyncio.Task.cancel()."""
    universe = load_tech_universe()
    if not universe:
        logger.error("Empty tech universe. Run `python -m fs_research_agent.tech_universe regenerate`.")
        return 1

    seen = _load_seen(data_root)
    if not seen:
        logger.info("No prior seen-accession state — bootstrapping from corpus on disk")
        seen = _bootstrap_seen_from_disk(data_root, universe)
        _save_seen(data_root, seen)
        total = sum(len(v) for v in seen.values())
        logger.info(f"Bootstrapped {total:,} accessions across {len(seen)} tickers")

    stop = asyncio.Event()

    if install_signal_handlers:
        def _handle_signal(signum, _frame):
            logger.info(f"received signal {signum} — shutting down after current cycle")
            stop.set()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _handle_signal)
            except ValueError:
                pass  # not in main thread

    cycle_num = 0
    while not stop.is_set():
        cycle_num += 1
        logger.info(f"━━━ Cycle {cycle_num} starting (universe={len(universe)} tickers, forms={list(forms)}) ━━━")
        try:
            stats = await run_one_cycle(
                universe=universe,
                data_root=data_root,
                forms=forms,
                seen=seen,
                keep_exhibits=keep_exhibits,
                max_age_days=max_age_days,
            )
            _save_cycle_stats(data_root, stats)
            logger.info(
                f"  cycle done in {stats.duration_s}s: "
                f"polled={stats.tickers_polled}, ticker_failures={stats.tickers_failed}, "
                f"new={stats.new_filings_found}, downloaded={stats.new_filings_downloaded}, "
                f"download_failures={stats.download_failures}"
            )
        except Exception as e:
            logger.exception(f"cycle {cycle_num} crashed: {e}")

        if once:
            break

        # Sleep, but wake immediately on signal
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_secs)
        except asyncio.TimeoutError:
            pass

    logger.info("watcher exiting cleanly")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", type=int, default=DEFAULT_POLL_INTERVAL_SECS,
                        help=f"Seconds between cycles (default: {DEFAULT_POLL_INTERVAL_SECS})")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                        help="Corpus root (default: main fs_research_agent corpus)")
    parser.add_argument("--forms", default=",".join(DEFAULT_FORMS),
                        help=f"Comma-separated forms to watch (default: {','.join(DEFAULT_FORMS)})")
    parser.add_argument("--no-exhibits", action="store_true", help="Skip exhibit downloading")
    parser.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS,
                        help=f"Only download filings filed in the last N days (default: {DEFAULT_MAX_AGE_DAYS}). "
                             f"Set to 0 to disable the age filter (will fetch everything missing — slow and noisy).")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit (use for cron)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "openai", "urllib3", "datamule"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    forms = [f.strip() for f in args.forms.split(",") if f.strip()]
    bad = [f for f in forms if f not in SUPPORTED_FORMS]
    if bad:
        parser.error(f"Unsupported forms: {bad}. Supported: {SUPPORTED_FORMS}")

    return asyncio.run(watcher_loop(
        interval_secs=args.interval,
        data_root=Path(args.data_root),
        forms=forms,
        keep_exhibits=not args.no_exhibits,
        once=args.once,
        max_age_days=args.max_age_days if args.max_age_days > 0 else None,
    ))


if __name__ == "__main__":
    sys.exit(main())
