"""
Tech-universe selector backed by US_TECH_CLEANED.json.

Filters the platform's tech master list down to tractable subsets and exposes
ticker lists for the batch ingest job.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

# US_TECH_CLEANED.json lives in the existing platform's data-ingestion folder.
_UNIVERSE_PATH = (
    Path(__file__).resolve().parents[1]
    / "agent" / "rag" / "data_ingestion" / "US_TECH_CLEANED.json"
)

_DEFAULT_SECTORS = ("Information Technology", "Communication Services")
_DEFAULT_MARKET_CAPS = ("Mega Cap", "Large Cap")


def load_universe(
    sectors: Iterable[str] = _DEFAULT_SECTORS,
    market_caps: Iterable[str] = _DEFAULT_MARKET_CAPS,
    universe_path: Optional[Path] = None,
) -> List[str]:
    """
    Return a deduped sorted list of tickers from US_TECH_CLEANED.json that
    match the given sector + market-cap filters.

    Defaults select the "Tier A" slice: 188 Mega+Large Cap IT/Comm Services
    tickers — the names you'd expect an analyst to actually reference.
    """
    path = Path(universe_path) if universe_path else _UNIVERSE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Universe file not found: {path}")

    sector_set = {s.strip() for s in sectors}
    cap_set = {c.strip() for c in market_caps}

    data = json.loads(path.read_text(encoding="utf-8"))
    companies = data.get("companies") or []
    seen: set[str] = set()
    out: List[str] = []
    for c in companies:
        sec = c.get("sector")
        cap = c.get("market_cap")
        ticker = (c.get("ticker") or "").upper()
        if not ticker or ticker in seen:
            continue
        if sectors and sec not in sector_set:
            continue
        if market_caps and cap not in cap_set:
            continue
        # Skip non-standard tickers (multi-class shares like AAPLC, AAPLD)
        if not ticker.isalpha():
            continue
        seen.add(ticker)
        out.append(ticker)
    out.sort()
    return out


def summarize_universe() -> str:
    tickers = load_universe()
    return f"Tech universe (Mega+Large IT/Comm): {len(tickers)} tickers"


if __name__ == "__main__":
    import sys
    tickers = load_universe()
    print(f"# {len(tickers)} tickers (Mega+Large Cap, IT + Communication Services)")
    for t in tickers:
        print(t)
