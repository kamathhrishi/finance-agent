#!/usr/bin/env python3
"""
Prototype: Explore 10-K exhibits for a single ticker.

Downloads one year of 10-K filings (including all exhibits) via the existing
download_and_extract_10k pipeline, then prints a structured breakdown of:

  1. Which exhibit types were found and how many chunks each contributes
  2. Sample content from each exhibit type
  3. How the existing `exhibit_source` / `exhibit_type` fields look
  4. A draft idea for how the RAG agent could select specific exhibits

Usage:
    python explore_10k_exhibits.py --ticker NET
    python explore_10k_exhibits.py --ticker AAPL --year 2023
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Make sure sibling modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent.parent / ".env", override=True)

from ingest_10k_filings_full import download_and_extract_10k


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, n: int = 300) -> str:
    text = text.replace("\n", " ").strip()
    return text[:n] + "…" if len(text) > n else text


def _exhibit_label(ex_type: str) -> str:
    """Human-readable description for common exhibit types."""
    labels = {
        "EX-13":   "Annual Report to Shareholders",
        "EX-21":   "List of Subsidiaries",
        "EX-23":   "Auditor Consent",
        "EX-23.1": "Auditor Consent",
        "EX-31.1": "CEO SOX 302 Certification",
        "EX-31.2": "CFO SOX 302 Certification",
        "EX-32.1": "CEO SOX 906 Certification",
        "EX-32.2": "CFO SOX 906 Certification",
        "EX-99.1": "Supplemental / Press Release",
        "EX-99.2": "Supplemental Financial Data",
        "EX-10":   "Material Contract",
        "EX-4":    "Instrument Defining Rights of Securities",
        "EX-3.1":  "Articles of Incorporation",
        "EX-3.2":  "Bylaws",
    }
    return labels.get(ex_type, "Other Exhibit")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_filing(ticker: str, fiscal_year: int, filing_data: dict) -> None:
    chunks = filing_data.get("hierarchical_chunks", [])
    doc_len = filing_data.get("document_length", 0)

    print(f"\n{'='*70}")
    print(f"  {ticker}  FY{fiscal_year}   |   {doc_len:,} chars total   |   {len(chunks)} chunks")
    print(f"{'='*70}")

    # --- Split chunks by source -----------------------------------------
    main_chunks: list[dict] = []
    exhibit_chunks: defaultdict[str, list[dict]] = defaultdict(list)

    for chunk in chunks:
        ex = chunk.get("exhibit_source")
        if ex:
            exhibit_chunks[ex].append(chunk)
        else:
            main_chunks.append(chunk)

    print(f"\n📄 Main 10-K body:  {len(main_chunks)} chunks")

    if not exhibit_chunks:
        print("\n  (no exhibits found in this filing)\n")
        return

    print(f"\n📎 Exhibits found ({len(exhibit_chunks)} types):\n")
    for ex_type in sorted(exhibit_chunks):
        ex_list = exhibit_chunks[ex_type]
        label   = _exhibit_label(ex_type)
        types   = set(c.get("type", "?") for c in ex_list)
        print(f"  {ex_type:12}  {label}")
        print(f"             {len(ex_list)} chunks  |  chunk types: {sorted(types)}")

    # --- Deep dive per exhibit ------------------------------------------
    for ex_type in sorted(exhibit_chunks):
        ex_list = exhibit_chunks[ex_type]
        label   = _exhibit_label(ex_type)
        print(f"\n{'─'*60}")
        print(f"  {ex_type} — {label}")
        print(f"{'─'*60}")

        # Show up to 3 sample chunks
        for i, chunk in enumerate(ex_list[:3], 1):
            path_str = " > ".join(chunk.get("path", []))
            content  = _truncate(chunk.get("content", ""))
            sec      = chunk.get("sec_section", "")
            print(f"\n  [chunk {i}]  path: {path_str or '(none)'}")
            if sec:
                print(f"             sec_section: {sec}")
            print(f"             content: {content}")

        if len(ex_list) > 3:
            print(f"\n  … and {len(ex_list) - 3} more chunks")

    # --- Draft: what metadata is available for agent exhibit selection --
    print(f"\n{'─'*60}")
    print("  DRAFT: exhibit_type values in chunk metadata")
    print("  (these are already stored in ten_k_chunks.exhibit_type)")
    print(f"{'─'*60}")
    sample_row = {
        "ticker":       ticker,
        "fiscal_year":  fiscal_year,
        "chunk_text":   "(chunk content here)",
        "exhibit_type": "EX-99.1",      # None for main body
        "sec_section":  "exhibit_99",
        "path_string":  "Exhibit (EX-99.1) > Press Release > Revenue",
    }
    print(json.dumps(sample_row, indent=4))

    print(f"\n{'─'*60}")
    print("  DRAFT: how the agent could filter / select exhibits")
    print(f"{'─'*60}")
    exhibit_index = [
        {
            "exhibit_type": ex_type,
            "label":        _exhibit_label(ex_type),
            "num_chunks":   len(clist),
            "sample":       _truncate(clist[0].get("content", ""), 120) if clist else "",
        }
        for ex_type, clist in sorted(exhibit_chunks.items())
    ]
    print("\n  Available exhibits (as a tool-call result the LLM could read):")
    for item in exhibit_index:
        print(f"    {item['exhibit_type']:12}  {item['label']:40}  ({item['num_chunks']} chunks)")
        print(f"               \"{item['sample']}\"")

    print(f"\n  Ideas for agent integration:")
    print("    A) Store exhibit_type in chunk metadata (DONE — ten_k_chunks.exhibit_type)")
    print("    B) Add exhibit_type filter to DB search query")
    print("       e.g.  WHERE exhibit_type = 'EX-99.1'  or  exhibit_type IS NULL (main body only)")
    print("    C) Give the LLM an exhibit index (types + descriptions) as part of the system prompt")
    print("       → LLM picks which exhibit types are relevant, search filters to those")
    print("    D) Treat each exhibit type as a separate 'section' in search routing")
    print("       (like we route 10-K sections vs. transcript chunks)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Explore exhibits in a 10-K filing")
    parser.add_argument("--ticker", required=True, help="Company ticker, e.g. NET")
    parser.add_argument("--year", type=int, default=None,
                        help="Fiscal year to inspect (default: most recent available)")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    # Download a single year — set start/end to the same year (or year-1 → year)
    if args.year:
        start_year = args.year
        end_year   = args.year
    else:
        from datetime import datetime
        end_year   = datetime.now().year
        start_year = end_year - 1          # grab at most 2 so we get at least one filing

    print(f"\n⬇️  Downloading 10-K for {ticker} ({start_year}–{end_year})…")
    filings_by_year = download_and_extract_10k(ticker, start_year, end_year)

    if not filings_by_year:
        print(f"❌  No filings found for {ticker} in {start_year}–{end_year}")
        sys.exit(1)

    # If user asked for a specific year, filter
    if args.year and args.year in filings_by_year:
        years_to_show = [args.year]
    else:
        years_to_show = sorted(filings_by_year.keys(), reverse=True)[:1]

    for fy in years_to_show:
        analyse_filing(ticker, fy, filings_by_year[fy])

    print(f"\n✅  Done.\n")


if __name__ == "__main__":
    main()
