#!/usr/bin/env python3
"""
Walk an existing corpus and apply table-column dedupe to every filing.md and
sections/*.md in place. Idempotent — re-running on already-cleaned files is a
no-op.

Usage:
    # Dry-run (count what would change)
    python -m agent.backfill_table_cleanup --dry-run

    # Apply to main corpus
    python -m agent.backfill_table_cleanup

    # Apply to benchmark corpus
    python -m agent.backfill_table_cleanup --data-root \
        agent/benchmarks/financebench/data
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from agent.ingest import DEFAULT_DATA_ROOT
from agent.markdown_cleanup import cleanup_markdown_tables


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    filings_root = root / "filings"
    if not filings_root.is_dir():
        print(f"❌ no filings dir at {filings_root}")
        return 1

    print(f"📁 Corpus: {filings_root}")
    print(f"   Mode: {'DRY-RUN' if args.dry_run else 'APPLY'}")
    print()

    # Targets: every filing.md plus every sections/*.md plus exhibits/*.md
    targets = []
    for p in filings_root.rglob("filing.md"):
        targets.append(p)
    for p in filings_root.rglob("sections/*.md"):
        targets.append(p)
    for p in filings_root.rglob("exhibits/*.md"):
        targets.append(p)
    print(f"Walking {len(targets):,} markdown files…")

    n_changed = 0
    n_unchanged = 0
    bytes_before = 0
    bytes_after = 0
    t0 = time.time()
    for i, p in enumerate(targets, 1):
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  ⚠ read failed {p}: {e}")
            continue
        if "|" not in src:
            n_unchanged += 1
            continue
        bytes_before += len(src)
        out = cleanup_markdown_tables(src)
        bytes_after += len(out)
        if out == src:
            n_unchanged += 1
            continue
        n_changed += 1
        if not args.dry_run:
            try:
                p.write_text(out, encoding="utf-8")
            except Exception as e:
                print(f"  ⚠ write failed {p}: {e}")
        if i % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{i:>6}/{len(targets):>6}]  changed={n_changed} unchanged={n_unchanged}  ({elapsed:.0f}s, {i/elapsed:.0f} files/s)")

    elapsed = time.time() - t0
    saved = bytes_before - bytes_after
    print()
    print(f"🏁 Done in {elapsed:.0f}s")
    print(f"   files changed:  {n_changed:,}")
    print(f"   files unchanged:{n_unchanged:,}")
    if bytes_before:
        print(f"   bytes saved:    {saved:,} ({saved*100//max(bytes_before,1)}% of touched files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
