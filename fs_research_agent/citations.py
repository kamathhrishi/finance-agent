"""
Extract citations from a FilesystemResearchAgent answer.

The agent is prompted to cite as `path:line` or `path:line-line`, e.g.

    Data center revenue grew to $115.2B (`filings/NVDA/10-K/FY2025/sections/item-7-mda.md:42`).

We walk the answer, find every such cite, deduplicate, and:
  1. Replace each occurrence with a marker `[FS-N]`
  2. Build a structured citation per unique cite, populated with metadata
     parsed from the path (ticker, filing_type, fiscal_year, section)
  3. Pull a short snippet from the file (for the citation card)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .highlight import extract_snippet


# Path shapes by form:
#   10-K:  filings/<TICKER>/10-K/FY####/(sections/|exhibits/)?<file>.md
#   10-Q:  filings/<TICKER>/10-Q/FY####/Q[1-4]/(sections/|exhibits/)?<file>.md
#   8-K:   filings/<TICKER>/8-K/YYYY-MM-DD/(sections/|exhibits/)?<file>.md
#
# The capture in `rest` covers everything after the year/quarter/date segment.
_PATH_RE = re.compile(
    r"filings/(?P<ticker>[A-Z][A-Z0-9._-]{0,9})/(?P<form>10-K|10-Q|8-K)/"
    r"(?P<period>FY?\d{4}(?:/Q[1-4])?|\d{4}-\d{2}-\d{2})/"
    r"(?P<rest>(?:sections/|exhibits/)?[A-Za-z0-9._\-]+\.md)"
)

# Full citation with optional line range. Tolerates surrounding backticks/parens.
# Captures: path  line_start  line_end?
_CITE_RE = re.compile(
    r"`?(filings/[A-Z][A-Z0-9._-]{0,9}/(?:10-K|10-Q|8-K)/"
    r"(?:FY?\d{4}(?:/Q[1-4])?|\d{4}-\d{2}-\d{2})/"
    r"(?:sections/|exhibits/)?[A-Za-z0-9._\-]+\.md)"
    r":(\d+)(?:-(\d+))?`?"
)

# Defensive cleanup: detect ABBREVIATED citation forms the LLM occasionally
# produces despite the prompt forbidding them. These never resolve to a real
# source so we strip them to avoid dead text in the UI.
#
# Matches things like:
#   (.../FY2017/...:37)
#   (.../FY2024/...:120-145)
#   `.../FY2025/sections/item-7-mda.md:42`
#   (item-7-mda:42)        ← stem-only abbreviation
_BROKEN_CITE_RES = [
    re.compile(r"\s*\(`?\.\.\.[^()`]*?:\d+(?:-\d+)?`?\)"),         # (.../FY####/...:NN)
    re.compile(r"\s*`\.\.\.[^`]*?:\d+(?:-\d+)?`"),                  # `.../...:NN`
    re.compile(r"\s*\(`?(?:item|section)[A-Za-z0-9._\-]*:\d+(?:-\d+)?`?\)"),  # (item-7-mda:42)
]

# Numbered-shorthand citations like "(1)", "(2, 3)", "(6, 7, 8)" that the
# model occasionally emits despite the prompt forbidding them. The frontend
# has no way to map "(2)" back to a source, so they render as literal dead
# text. Strip them ONLY when the entire parenthetical is small numbers
# (1-3 digit) optionally separated by commas — never strip "(2024)" or
# "(45 employees)" etc.
#
# Pattern detail:
#   \s*           — drop leading whitespace
#   \(            — open paren
#   \d{1,3}       — first number (small)
#   (\s*,\s*\d{1,3}){0,9}  — up to 9 more comma-separated small numbers
#   \)            — close paren
#   (?=...)       — followed by sentence punctuation, end-of-string, or another bracketed citation
_BARE_NUMBERED_CITE_RE = re.compile(
    # Lookahead now also accepts `|` and newlines so this catches `(3, 4, 3)`
    # right before a markdown table cell delimiter or end-of-line.
    r"\s*\(\d{1,3}(?:\s*,\s*\d{1,3}){0,9}\)(?=[\s.,;:!?\)|\n]|$|\[)"
)

# Bracketless labeled citation tokens like `8K-1`, `10K-3`, `10Q-2`, `FS-4`
# that may appear inside parens (`(8K-1, 8K-2)`) or mixed with bare numbers
# (`(1, 2, 8K-1)`). The model sometimes emits these instead of the canonical
# square-bracketed form `[8K-1]` — we normalize both cases below.
_LABELED_CITE_TOKEN_RE = re.compile(r"^(?:10K|10Q|8K|FS)-\d+$", re.IGNORECASE)

# Defensive: strip any leftover RAW corpus paths the model emitted without a
# line number. These don't match _CITE_RE (which requires `:line`) so they
# survive extraction and render as user-visible filesystem paths — a privacy
# leak ("filings/NET/10-K/FY2025/sections/item-1-business.md" should never
# show up in the user's answer).
_RAW_PATH_RES = [
    # Markdown — without :line (canonical form WITH :line gets extracted
    # to [10K-N] markers above; this catches the leftover incomplete ones).
    re.compile(r"\s*\(`?filings/[A-Z][A-Z0-9._-]{0,9}/(?:10-K|10-Q|8-K)/[^()`]+?\.md`?\)"),
    re.compile(r"`filings/[A-Z][A-Z0-9._-]{0,9}/(?:10-K|10-Q|8-K)/[^`]+?\.md`"),
    re.compile(r"(?<![:`/\w])filings/[A-Z][A-Z0-9._-]{0,9}/(?:10-K|10-Q|8-K)/[A-Za-z0-9./_-]+?\.md(?!\:)"),

    # Non-markdown corpus paths — `metadata.json:N`, `INDEX.md:N`, etc.
    # The agent reads these for its own use (filing date discovery, index
    # browsing) but they should NEVER appear as user-visible citations —
    # they're agent-internal scratchpad. Strip both with-line and without.
    re.compile(r"\s*\(`?filings/[A-Z][A-Z0-9._-]{0,9}/(?:10-K|10-Q|8-K)/[^()`]+?\.json(?::\d+(?:-\d+)?)?`?\)"),
    re.compile(r"\s*\(`?filings/[A-Z][A-Z0-9._-]{0,9}/INDEX\.md(?::\d+(?:-\d+)?)?`?\)"),
    re.compile(r"\s*\(`?filings/[A-Z][A-Z0-9._-]{0,9}/(?:10-K|10-Q|8-K)/[^()`]+?/INDEX\.md(?::\d+(?:-\d+)?)?`?\)"),
]


# Map raw section keys (file stems) to human-readable subtitles for the
# citation card. These match the SEC item naming conventions and stay generic
# across companies. Falls back to a Title-Cased version of the stem if missing.
_SECTION_LABELS: Dict[str, str] = {
    # ── 10-K items ────────────────────────────────────────────────────────────
    "item-1-business": "Business overview (Item 1)",
    "item-1a-risk-factors": "Risk Factors (Item 1A)",
    "item-1b-unresolved-staff-comments": "Unresolved Staff Comments (Item 1B)",
    "item-1c-cybersecurity": "Cybersecurity (Item 1C)",
    "item-2-properties": "Properties (Item 2)",
    "item-3-legal-proceedings": "Legal Proceedings (Item 3)",
    "item-4-mine-safety": "Mine Safety (Item 4)",
    "item-5-market-for-registrants-equity": "Market for Registrant's Equity (Item 5)",
    "item-6-selected-financial-data": "Selected Financial Data (Item 6)",
    "item-7-mda": "MD&A (Item 7)",
    "item-7a-quant-qual-disclosures": "Market Risk (Item 7A)",
    "item-8-financial-statements": "Financial Statements (Item 8)",
    "item-9-changes-in-accountants": "Changes in Accountants (Item 9)",
    "item-9a-controls-and-procedures": "Controls & Procedures (Item 9A)",
    "item-9b-other-information": "Other Information (Item 9B)",
    "item-10-directors-and-officers": "Directors & Officers (Item 10)",
    "item-11-executive-compensation": "Executive Compensation (Item 11)",
    "item-12-security-ownership": "Security Ownership (Item 12)",
    "item-13-related-party-transactions": "Related Party Transactions (Item 13)",
    "item-14-principal-accountant-fees": "Principal Accountant Fees (Item 14)",
    "item-15-exhibits": "Exhibits (Item 15)",
    # ── 10-Q Part I items ─────────────────────────────────────────────────────
    "item-1-financial-statements": "Financial Statements (10-Q Part I, Item 1)",
    "item-2-mda": "MD&A (10-Q Part I, Item 2)",
    "item-3-quant-qual-market-risk": "Market Risk (10-Q Part I, Item 3)",
    "item-4-controls-and-procedures": "Controls & Procedures (10-Q Part I, Item 4)",
    # ── 10-Q Part II items ────────────────────────────────────────────────────
    "item-2-unregistered-equity-sales": "Unregistered Equity Sales (10-Q Part II, Item 2)",
    "item-3-defaults-on-senior-securities": "Defaults on Senior Securities (10-Q Part II, Item 3)",
    "item-5-other-information": "Other Information (10-Q Part II, Item 5)",
    "item-6-exhibits": "Exhibits (10-Q Part II, Item 6)",
    # ── 8-K items (decimal-numbered) ──────────────────────────────────────────
    "item-1-01-material-definitive-agreement": "Material Definitive Agreement (8-K Item 1.01)",
    "item-1-02-termination-material-agreement": "Agreement Termination (8-K Item 1.02)",
    "item-1-03-bankruptcy": "Bankruptcy (8-K Item 1.03)",
    "item-2-01-acquisition-disposition": "Acquisition/Disposition (8-K Item 2.01)",
    "item-2-02-results-of-operations": "Results of Operations (8-K Item 2.02)",
    "item-2-03-direct-financial-obligation": "Direct Financial Obligation (8-K Item 2.03)",
    "item-2-04-triggering-events": "Triggering Events (8-K Item 2.04)",
    "item-2-05-exit-or-disposal": "Exit/Disposal Activities (8-K Item 2.05)",
    "item-2-06-material-impairments": "Material Impairments (8-K Item 2.06)",
    "item-3-01-delisting": "Delisting (8-K Item 3.01)",
    "item-3-02-unregistered-equity-sales": "Unregistered Equity Sales (8-K Item 3.02)",
    "item-3-03-rights-modification": "Rights Modification (8-K Item 3.03)",
    "item-4-01-accountant-change": "Accountant Change (8-K Item 4.01)",
    "item-4-02-non-reliance-on-financials": "Restatement (8-K Item 4.02)",
    "item-5-01-change-in-control": "Change in Control (8-K Item 5.01)",
    "item-5-02-officer-departure-election": "Officer Departure/Election (8-K Item 5.02)",
    "item-5-03-bylaw-amendments": "Bylaw Amendments (8-K Item 5.03)",
    "item-5-07-shareholder-vote": "Shareholder Vote (8-K Item 5.07)",
    "item-5-08-shareholder-nominations": "Shareholder Nominations (8-K Item 5.08)",
    "item-7-01-regulation-fd": "Regulation FD (8-K Item 7.01)",
    "item-8-01-other-events": "Other Events (8-K Item 8.01)",
    "item-9-01-financial-statements-and-exhibits": "Financial Statements & Exhibits (8-K Item 9.01)",
}


# Map SEC form → marker prefix used inline. The frontend regex in
# ChatMessage.preprocessCitationMarkers must accept all of these for the
# inline `[X-N]` markers to render as clickable hyperlinks.
_MARKER_PREFIX: Dict[str, str] = {
    "10-K": "10K",
    "10-Q": "10Q",
    "8-K":  "8K",
}


@dataclass
class FsCitation:
    # User-facing marker, e.g. "[10K-3]", "[10Q-2]", "[8K-1]".
    # Format matches the platform's existing inline-marker convention so the
    # frontend regex turns it into a hyperlink (see ChatMessage.tsx
    # preprocessCitationMarkers — that regex must include 8K-?\d+).
    marker: str
    chunk_id: str                  # marker minus brackets, e.g. "10K-3"
    path: str                      # relative to corpus root
    line_start: int
    line_end: int
    chunk_text: str                # snippet pulled from the file (preview)
    ticker: Optional[str] = None
    filing_type: Optional[str] = None
    fiscal_year: Optional[int] = None
    section: Optional[str] = None  # human-readable, e.g. "MD&A (Item 7)"
    type: str = "10-K"             # citation type (drives frontend badge + getCitationType)
    citation_type: str = "10-K"
    source_backend: str = "fs_research"
    relevance_score: float = 0.9

    def to_chat_citation(self) -> Dict[str, object]:
        d: Dict[str, object] = {k: v for k, v in asdict(self).items() if v is not None}
        return d


def _parse_path_meta(path: str) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
    """Pull ticker / form / fiscal_year / section (humanized) from a corpus-relative path."""
    m = _PATH_RE.match(path)
    if not m:
        return None, None, None, None
    ticker = m.group("ticker")
    form = m.group("form")
    period = m.group("period")  # `FY####`, `FY####/Q#`, or `YYYY-MM-DD`
    # Pull a 4-digit year from whichever period style we got
    fy_digits = re.search(r"\d{4}", period or "")
    fy = int(fy_digits.group()) if fy_digits else None
    rest = m.group("rest")
    section = None
    if rest.startswith("sections/"):
        stem = Path(rest).stem  # e.g. "item-7-mda"
        section = _SECTION_LABELS.get(stem) or stem.replace("-", " ").title()
    elif rest.startswith("exhibits/"):
        # e.g. "EX-99.1.md" -> "Exhibit EX-99.1"
        section = f"Exhibit {Path(rest).stem}"
    return ticker, form, fy, section


def _read_file_lines(corpus_root: Path, rel_path: str) -> Optional[str]:
    """Safe-read a file inside the corpus; returns None if outside or missing."""
    try:
        target = (corpus_root / rel_path).resolve()
        # Containment check
        target.relative_to(corpus_root.resolve())
    except (ValueError, OSError):
        return None
    if not target.is_file():
        return None
    try:
        return target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def extract_citations(
    answer: str,
    corpus_root: Path,
) -> Tuple[str, List[FsCitation]]:
    """
    Scan `answer` for `path:line(-line)` citations. Return:
      - the answer with each cite replaced by `[FS-N]` (in order of first appearance)
      - the list of unique structured citations, in the same order

    Two cites to the same `(path, line_start, line_end)` triple share a marker.
    """
    citations: List[FsCitation] = []
    key_to_marker: Dict[Tuple[str, int, int], str] = {}

    # Walk all matches in order; build a list of (span, replacement)
    spans: List[Tuple[int, int, str]] = []
    for m in _CITE_RE.finditer(answer):
        path = m.group(1)
        line_start = int(m.group(2))
        line_end = int(m.group(3)) if m.group(3) else line_start
        if line_end < line_start:
            line_end = line_start

        key = (path, line_start, line_end)
        if key in key_to_marker:
            marker = key_to_marker[key]
        else:
            ticker, form, fy, section = _parse_path_meta(path)
            # Form-aware marker prefix so 10-K / 10-Q / 8-K cites are visually
            # distinct in the chat UI ("[10Q-3]" vs "[8K-1]").
            prefix = _MARKER_PREFIX.get(form or "10-K", "10K")
            # Number per-prefix so each form gets its own running sequence
            seq = sum(1 for c in citations if c.marker.startswith(f"[{prefix}-")) + 1
            chunk_id = f"{prefix}-{seq}"
            marker = f"[{chunk_id}]"
            key_to_marker[key] = marker

            file_text = _read_file_lines(corpus_root, path) or ""
            # Pad the snippet a little so the citation card preview shows
            # context, not just the bare cited line(s).
            pad_start = max(1, line_start - 2)
            pad_end = line_end + 2
            snippet = (
                extract_snippet(file_text, pad_start, pad_end, max_chars=320)
                if file_text else ""
            )

            citations.append(
                FsCitation(
                    marker=marker,
                    chunk_id=chunk_id,
                    path=path,
                    line_start=line_start,
                    line_end=line_end,
                    chunk_text=snippet,
                    ticker=ticker,
                    filing_type=form,
                    fiscal_year=fy,
                    section=section,
                    # Map all SEC form types to the '10-K' citation type for now —
                    # frontend's getCitationType groups them into the same '10k' badge
                    # bucket, which is what we want (single "Filings" pane).
                    type=form or "10-K",
                    citation_type=form or "10-K",
                )
            )

        spans.append((m.start(), m.end(), marker))

    # Replace right-to-left so earlier offsets stay valid
    out = answer
    for start, end, replacement in sorted(spans, key=lambda s: -s[0]):
        out = out[:start] + replacement + out[end:]

    # Defensive: strip any leftover abbreviated citation patterns the LLM
    # may have emitted despite the prompt — they would render as dead text.
    for pat in _BROKEN_CITE_RES:
        out = pat.sub("", out)

    # Defensive: strip any raw corpus paths the LLM emitted WITHOUT a line
    # number (those wouldn't match _CITE_RE above). User must never see
    # filesystem paths in the rendered answer.
    for pat in _RAW_PATH_RES:
        out = pat.sub("", out)

    # Strip bare numbered-shorthand citations like "(1)", "(2, 3)", "(6, 7, 8)".
    # Run UNCONDITIONALLY — previously gated on `if citations:` (i.e. at least
    # one real citation extracted), but that gating fails when the model only
    # emits non-extractable paths like `metadata.json:N`: extractor returns
    # 0 citations, the bare-number stripper never fires, and `(1)`...`(34)`
    # leak through naked. We accept the trade-off that legitimate prose like
    # "1980s (1)" might lose its parenthetical — that pattern is much rarer
    # in financial-research output than the bare-number citation footgun.
    out = _BARE_NUMBERED_CITE_RE.sub("", out)

    if citations:

        # Normalise EVERY parenthetical that looks like a citation group.
        # Three cases the model emits despite the prompt:
        #   (a) bracketless labeled refs:  "(8K-1, 8K-2)"          → "[8K-1] [8K-2]"
        #   (b) mixed bracketed + bare:    "(3, [8K-1])"           → "([8K-1])"
        #   (c) mixed bracketless + bare:  "(1, 2, 8K-1)"          → "[8K-1]"
        # If a parenthetical has neither a [bracketed] nor a labeled token,
        # we leave it alone — it might be legitimate prose like "(see below)".
        def _scrub_mixed(m: re.Match) -> str:
            inner = m.group(1)
            parts = [p.strip() for p in inner.split(",")]

            # Drop bare numeric-only tokens
            non_bare = [p for p in parts if not re.fullmatch(r"\d{1,3}", p)]
            if not non_bare:
                return m.group(0)  # untouched — might be legit prose

            # Promote any bracketless labeled token to the canonical [LBL-N] form
            normalised: List[str] = []
            saw_label = False
            for tok in non_bare:
                if _LABELED_CITE_TOKEN_RE.match(tok):
                    normalised.append(f"[{tok.upper()}]")
                    saw_label = True
                elif tok.startswith("[") and tok.endswith("]"):
                    normalised.append(tok)
                    saw_label = True
                else:
                    normalised.append(tok)

            if not saw_label:
                return m.group(0)  # nothing recognisable — leave untouched

            # Spaces between adjacent bracket markers reads cleaner than commas
            return "(" + " ".join(normalised) + ")"

        out = re.sub(r"\(([^()]+)\)", _scrub_mixed, out)

    return out, citations
