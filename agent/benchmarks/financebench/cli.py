#!/usr/bin/env python3
"""
End-to-end CLI for the FinanceBench benchmark on agent.

Common flows:

    # Full smoke test on 5 questions, no download (must already have filings)
    python -m agent.benchmarks.financebench.cli run --limit 5

    # Ensure required filings are present, then run all 136 questions
    python -m agent.benchmarks.financebench.cli download
    python -m agent.benchmarks.financebench.cli run

    # Download just for one company subset, then run that subset
    python -m agent.benchmarks.financebench.cli download --companies 3M,AMD,Adobe
    python -m agent.benchmarks.financebench.cli run --companies 3M,AMD,Adobe

    # Resume — re-running with the same `--run-name` skips already-judged questions
    python -m agent.benchmarks.financebench.cli run --run-name nano_2026-05-01

    # See what the corpus is missing without downloading anything
    python -m agent.benchmarks.financebench.cli download --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import List, Optional

# Ensure SEC user-agent is set before any datamule imports happen downstream
os.environ.setdefault("DATAMULE_SEC_USER_AGENT", "StrataLens kamathhrishi@gmail.com")


def _split_companies(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [c.strip() for c in s.split(",") if c.strip()]


def cmd_download(args: argparse.Namespace) -> int:
    from .download import ensure_required_filings
    summary = ensure_required_filings(
        only_companies=_split_companies(args.companies),
        keep_exhibits=not args.no_exhibits,
        dry_run=args.dry_run,
    )
    print("\n━━━ DOWNLOAD SUMMARY ━━━")
    for ticker in sorted(summary):
        s = summary[ticker]
        print(f"  {ticker:6}  needed={s['needed']:>3}  had={s['had']:>3}  downloaded={s['downloaded']:>3}  still_missing={s['still_missing']:>3}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import run_benchmark

    run_name = args.run_name or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    asyncio.run(run_benchmark(
        run_name=run_name,
        only_companies=_split_companies(args.companies),
        limit=args.limit,
        concurrency=args.concurrency,
        max_tool_calls=args.max_tool_calls,
        agent_model=args.agent_model,
        judge_model=args.judge_model,
        resume=not args.no_resume,
    ))
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    from pathlib import Path
    from .runner import _summarize, RESULTS_ROOT, PASS_THRESHOLD
    import json
    p = RESULTS_ROOT / args.run_name / "results.jsonl"
    if not p.is_file():
        print(f"No results at {p}")
        return 1
    s = _summarize(p)
    (p.parent / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(json.dumps(s, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="Ensure FinanceBench-required filings are in the corpus")
    p_dl.add_argument("--companies", default="", help="Comma-separated subset of FinanceBench company names")
    p_dl.add_argument("--no-exhibits", action="store_true")
    p_dl.add_argument("--dry-run", action="store_true")
    p_dl.set_defaults(func=cmd_download)

    p_run = sub.add_parser("run", help="Run the agent against FinanceBench and judge each answer")
    p_run.add_argument("--companies", default="", help="Subset to run")
    p_run.add_argument("--limit", type=int, default=None, help="Cap number of questions (smoke testing)")
    p_run.add_argument("--concurrency", type=int, default=4, help="Concurrent agent runs in flight (default: 4)")
    p_run.add_argument("--max-tool-calls", type=int, default=25, help="Per-question agent tool budget")
    p_run.add_argument("--agent-model", default=None, help="Override agent model (default: gpt-5.4-mini)")
    p_run.add_argument("--judge-model", default=None, help="Override judge model (default: gpt-5.4-mini)")
    p_run.add_argument("--run-name", default=None, help="Output dir name; resume by reusing the same name")
    p_run.add_argument("--no-resume", action="store_true", help="Do not skip questions already in results.jsonl")
    p_run.set_defaults(func=cmd_run)

    p_sum = sub.add_parser("summary", help="Recompute summary.json from a run's results.jsonl")
    p_sum.add_argument("run_name")
    p_sum.set_defaults(func=cmd_summary)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet datamule's noisy progress logs unless verbose
    for noisy in ("httpx", "openai", "urllib3", "datamule"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
