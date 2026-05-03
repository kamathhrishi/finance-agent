"""
Canonical tech universe persisted as `fs_research_agent/tech_universe.json`.

Why a separate file from `universe.py`:
  - `universe.py` runs a filter over `US_TECH_CLEANED.json` every time and
    pulls in noisy entries (class shares like AAPLC/AAPLD/AVGOP, delisted
    tickers like ATVI/SQ, and a few ETF-like entries).
  - The watcher needs a stable, deduped, CIK-resolved list so it can poll
    SEC EDGAR's submissions endpoint reliably.
  - You want to edit the universe by hand sometimes (add a ticker not in
    the source dataset, exclude one).

This module:
  - regenerates the canonical JSON from US_TECH_CLEANED.json + SEC's
    company_tickers.json, with conservative filtering applied
  - exposes `load_tech_universe()` returning a list of TickerSpec entries
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import httpx

logger = logging.getLogger("fs_research_agent.tech_universe")

PKG_ROOT = Path(__file__).resolve().parent
TECH_UNIVERSE_PATH = PKG_ROOT / "tech_universe.json"
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


# Manually excluded — ticker no longer resolvable in SEC's company_tickers map
# OR known to map to a delisted filer that produces no current filings.
HARD_EXCLUDE: Set[str] = {
    "ATVI",   # Activision Blizzard — acquired by MSFT Oct 2023
    "SQ",     # Block — renamed to XYZ in 2024
    "AVLR",   # Avalara — taken private 2022
    "ABMD",   # Abiomed — acquired by JNJ 2022
    "VMW",    # VMware — acquired by AVGO 2023
    "TWTR",   # Twitter — taken private 2022 (renamed X)
    "SPLK",   # Splunk — acquired by Cisco 2024
}

# Known-good additions — companies that the FinanceDatabase filter would miss
# but obviously belong in a "tech" universe. Three buckets:
#   1. Recent IPOs not yet in FinanceDatabase's quarterly snapshot
#   2. Foreign-domiciled US-listed companies (ADRs / 20-F filers) — some will
#      fail SEC CIK resolution if they don't file with SEC at all (Shopify,
#      Spotify) and get dropped silently
#   3. Tech-by-business-model but not by GICS sector (AMZN in Consumer
#      Discretionary, V/MA/PYPL in Financials, exchanges/data in Capital
#      Markets, EV makers in Automobiles)
# Each entry is `ticker → display name`. The display name is overwritten by
# whatever SEC's company_tickers.json returns at CIK-resolution time, so the
# values here are just for documentation / traceability.
EXTRA_TICKERS: Dict[str, str] = {
    # ── Recent IPOs not yet in FinanceDatabase quarterly snapshot ──
    "FIG":  "Figma",
    "RDDT": "Reddit",
    "KVYO": "Klaviyo",
    "PCOR": "Procore",
    "TOST": "Toast",
    "CHYM": "Chime",
    # ── Foreign-domiciled US-listed (most file 20-F; will fail CIK lookup) ──
    "ARM":  "Arm Holdings",
    "MNDY": "Monday.com",
    "SHOP": "Shopify",
    "SPOT": "Spotify",
    "BABA": "Alibaba",
    "JD":   "JD.com",
    "BIDU": "Baidu",
    "NTES": "NetEase",
    # ── E-commerce / digital marketplaces (GICS: Consumer Discretionary) ──
    "AMZN": "Amazon", "EBAY": "eBay", "ETSY": "Etsy", "CHWY": "Chewy",
    "W":    "Wayfair", "CVNA": "Carvana", "MELI": "MercadoLibre",
    # ── Payment networks / payments infra (GICS: Financials/Consumer Finance) ──
    "V":    "Visa", "MA": "Mastercard", "PYPL": "PayPal",
    "FI":   "Fiserv", "FIS": "Fidelity National Information Services",
    # ── Exchanges + financial data (GICS: Financials/Capital Markets) ──
    "CME":  "CME Group", "ICE": "Intercontinental Exchange",
    "NDAQ": "Nasdaq Inc", "CBOE": "Cboe Global Markets",
    "MKTX": "MarketAxess", "SPGI": "S&P Global", "MCO": "Moody's",
    "MSCI": "MSCI Inc", "FDS": "FactSet", "MORN": "Morningstar",
    # ── Digital brokers (GICS: Financials/Capital Markets) ──
    "HOOD": "Robinhood", "IBKR": "Interactive Brokers", "SCHW": "Charles Schwab",
    # ── Fintech mid-caps (GICS: Financials, but pure software businesses) ──
    "SOFI": "SoFi Technologies", "UPST": "Upstart Holdings",
    # ── EV / hardware-as-tech (GICS: Consumer Discretionary/Automobiles) ──
    "TSLA": "Tesla", "RIVN": "Rivian", "LCID": "Lucid",
    # ── Tech-data hiding in 'Professional Services' (GICS: Industrials) ──
    "ADP":  "Automatic Data Processing", "EFX": "Equifax",
    "VRSK": "Verisk Analytics", "CPRT": "Copart",
    # ── Health-tech (GICS: Health Care, cherry-picked from a noisy industry) ──
    "VEEV": "Veeva Systems", "DOCS": "Doximity", "TDOC": "Teladoc",
    # ── Notable majors that FinanceDatabase's snapshot-date / sector logic
    #    happens to miss but every analyst expects to be in any tech universe
    "PANW": "Palo Alto Networks",       # cybersecurity, ~$120B
    "ANET": "Arista Networks",          # cloud networking, ~$100B
    "NXPI": "NXP Semiconductors",       # semis, ~$60B (Dutch domicile, US-listed)
    "TEAM": "Atlassian",                # SaaS, ~$30B (Australian domicile, US-listed)
    "TTWO": "Take-Two Interactive",     # gaming, GTA publisher
}


@dataclass
class TickerSpec:
    ticker: str
    cik: str            # 10-digit zero-padded
    company_name: str

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# SEC ticker → CIK resolution
# ──────────────────────────────────────────────────────────────────────────────


def _fetch_sec_ticker_map(user_agent: str) -> Dict[str, dict]:
    """Returns {TICKER: {cik_str, ticker, title}} from SEC."""
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    r = httpx.get(SEC_TICKER_MAP_URL, headers=headers, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    out: Dict[str, dict] = {}
    for entry in data.values():
        t = (entry.get("ticker") or "").upper()
        if t:
            out[t] = entry
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Build / load
# ──────────────────────────────────────────────────────────────────────────────


_PRIMARY_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


def _is_primary_ticker(t: str) -> bool:
    """Drop class-share variants (AAPLC, AAPLD, AVGOP) and oddballs."""
    if not _PRIMARY_TICKER_RE.match(t):
        return False
    # Heuristic: tickers ending with single trailing C / D / P / W /
    # numeric suffix are usually class shares of a base ticker. We only
    # drop them if a shorter base ticker IS in the same universe.
    # That check is done by the caller.
    return True


# US primary-listing exchange codes used by FinanceDatabase. Anything outside
# this set (PNK / OTC pink sheets, foreign exchanges like FRA / LSE / MEX /
# SAO / etc.) is either a duplicate cross-listing of a US company or an OTC
# stock that doesn't file with the SEC the way our agent expects.
US_PRIMARY_EXCHANGES = {
    "NMS",  # NASDAQ Global Select Market
    "NYQ",  # New York Stock Exchange
    "NCM",  # NASDAQ Capital Market
    "NGM",  # NASDAQ Global Market
    "ASE",  # NYSE American (formerly AMEX)
}


def _load_candidates_from_financedatabase(
    sectors: List[str],
    market_caps: List[str],
) -> List[str]:
    """Pull US-listed tech tickers from the `financedatabase` package.

    FinanceDatabase (https://github.com/jerbouma/FinanceDatabase) ships a
    quarterly-refreshed snapshot of ~150k global equities with sector /
    industry / market_cap / country / exchange metadata. Replaces the
    older static US_TECH_CLEANED.json which had to be hand-refreshed.

    Filter cascade:
      1. country = United States (catches US-domiciled equities)
      2. sector ∈ {sectors}
      3. market_cap ∈ {market_caps}
      4. exchange ∈ US_PRIMARY_EXCHANGES — CRITICAL: without this we'd
         pull every foreign cross-listing (e.g. MSFTC.BA on Buenos Aires,
         AAPL.MX on Mexico, all the same underlying companies). The
         country='United States' filter alone returns ~5× more rows than
         actual US primary listings because FinanceDatabase indexes each
         exchange separately.

    CIK resolution and class-share filtering happen downstream in the caller.
    """
    try:
        import financedatabase as fd
    except ImportError as e:
        raise RuntimeError(
            "financedatabase is required for universe regeneration. "
            "Install with: pip install financedatabase"
        ) from e

    eq = fd.Equities()
    df = eq.select(country="United States", sector=sectors)
    if market_caps:
        df = df[df["market_cap"].isin(market_caps)]
    df = df[df["exchange"].isin(US_PRIMARY_EXCHANGES)]
    # The dataframe index is the symbol/ticker. Drop empties.
    tickers = sorted({str(t).upper() for t in df.index if t and isinstance(t, str)})
    return tickers


def regenerate_tech_universe(
    *,
    user_agent: str,
    sectors: Optional[List[str]] = None,
    market_caps: Optional[List[str]] = None,
    save: bool = True,
) -> List[TickerSpec]:
    """Build the canonical universe and (optionally) persist to JSON.

    Defaults: Mega+Large Cap × IT + Communication Services pulled from
    `financedatabase`, deduped, with class shares dropped, plus
    HARD_EXCLUDE removed and EXTRA_TICKERS added. Each entry's CIK is
    resolved via SEC's ticker map; entries with no resolvable CIK are
    dropped.
    """
    # Step 1: candidate set from FinanceDatabase
    sectors = sectors or ["Information Technology", "Communication Services"]
    market_caps = market_caps or ["Mega Cap", "Large Cap"]
    candidates = _load_candidates_from_financedatabase(sectors, market_caps)
    candidates = [t for t in candidates if _is_primary_ticker(t)]
    candidates_set: Set[str] = set(candidates)

    # Drop apparent class-share variants when the base ticker is also present
    # (e.g. drop AAPLC, AAPLD if AAPL is in the set).
    deduped: List[str] = []
    for t in candidates:
        if len(t) > 4:
            base = t[:-1]
            if base in candidates_set:
                continue
        deduped.append(t)
    candidates = deduped

    # Apply hard excludes + extras
    candidates = [t for t in candidates if t not in HARD_EXCLUDE]
    for extra in EXTRA_TICKERS:
        if extra not in candidates and extra not in HARD_EXCLUDE:
            candidates.append(extra)

    # Step 2: resolve CIKs via SEC
    print(f"Resolving CIKs for {len(candidates)} tickers from SEC…")
    sec_map = _fetch_sec_ticker_map(user_agent)
    out: List[TickerSpec] = []
    missing: List[str] = []
    for t in sorted(set(candidates)):
        entry = sec_map.get(t)
        if not entry:
            missing.append(t)
            continue
        cik = str(entry.get("cik_str", "")).zfill(10)
        name = entry.get("title", t)
        out.append(TickerSpec(ticker=t, cik=cik, company_name=name))

    if missing:
        print(f"  ⚠ {len(missing)} ticker(s) had no SEC CIK and were dropped: {', '.join(missing[:15])}{' …' if len(missing) > 15 else ''}")

    if save:
        payload = {
            "meta": {
                "generated_from": "financedatabase (Equities) + SEC company_tickers.json",
                "sectors": sectors,
                "market_caps": market_caps,
                "ticker_count": len(out),
                "hard_excluded": sorted(HARD_EXCLUDE),
            },
            "tickers": [t.to_dict() for t in out],
        }
        TECH_UNIVERSE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"✅ Wrote {len(out)} tickers to {TECH_UNIVERSE_PATH}")

    return out


def load_tech_universe() -> List[TickerSpec]:
    """Load the persisted canonical universe. Caller must have run
    `regenerate_tech_universe()` at least once."""
    if not TECH_UNIVERSE_PATH.is_file():
        raise FileNotFoundError(
            f"{TECH_UNIVERSE_PATH} does not exist. Run "
            f"`python -m fs_research_agent.tech_universe regenerate` first."
        )
    data = json.loads(TECH_UNIVERSE_PATH.read_text(encoding="utf-8"))
    return [TickerSpec(**t) for t in data.get("tickers", [])]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _main_cli() -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Manage the canonical tech universe.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_regen = sub.add_parser("regenerate", help="Rebuild tech_universe.json from sources")
    p_regen.add_argument("--user-agent", default=os.getenv("DATAMULE_SEC_USER_AGENT", "StrataLens kamathhrishi@gmail.com"))
    p_regen.add_argument(
        "--include-mid-cap",
        action="store_true",
        help="Include Mid Cap companies in addition to Mega + Large (~2× growth, mostly mid-cap software/semis).",
    )

    p_show = sub.add_parser("show", help="List tickers in the persisted universe")

    args = parser.parse_args()
    if args.cmd == "regenerate":
        market_caps = ["Mega Cap", "Large Cap"]
        if getattr(args, "include_mid_cap", False):
            market_caps.append("Mid Cap")
        regenerate_tech_universe(
            user_agent=args.user_agent,
            market_caps=market_caps,
        )
    elif args.cmd == "show":
        ts = load_tech_universe()
        print(f"{len(ts)} tickers in {TECH_UNIVERSE_PATH}")
        for t in ts:
            print(f"  {t.ticker:<6}  CIK={t.cik}  {t.company_name}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main_cli())
