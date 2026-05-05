# Research corpus

SEC filings (10-K, 10-Q, 8-K) plus filtered material exhibits, organized for
filesystem-based research. The agent should read this file and `INDEX.md`
before grepping.

## Layout

```
filings/<TICKER>/
    INDEX.md                                   ← per-ticker index of every filing
    10-K/<FY-LABEL>/                           ← e.g. FY2025
        filing.md
        metadata.json
        sections/<item>.md                     ← when section parsing succeeds
        exhibits/EX-<n>.md                     ← filtered substantive exhibits
    10-Q/<FY-LABEL>/<QUARTER>/                 ← e.g. FY2025/Q3
        (same internal layout)
    8-K/<YYYY-MM-DD>/                          ← keyed by filing date (events)
        (same internal layout)
```

- `<TICKER>` is the company stock ticker (uppercase).
- `<FY-LABEL>` is `FY<YYYY>` derived from the filing's period of report.
- `<QUARTER>` is `Q1` / `Q2` / `Q3` / `Q4` from the **calendar** quarter of the
  period-end month. Note: companies with non-Dec fiscal years (e.g. NVDA's
  fiscal year ends in late January) will have calendar quarters that don't
  match their fiscal quarters. Always confirm by reading `metadata.json`.

## How to navigate efficiently

1. Read top-level `INDEX.md` to see what tickers are available and which forms
   each has.
2. Read `filings/<TICKER>/INDEX.md` to see every filing for one ticker. This
   file is small even for tickers with many filings.
3. Read a filing's `metadata.json` to confirm filing date, fiscal label, parsed
   section keys, and the list of exhibits.
4. Prefer reading specific sections or exhibits rather than the full
   `filing.md`. Use `grep` to localize before reading.
5. If `sections/` is empty (parsing fell back), use `grep` over `filing.md`.

## Form-specific tips

- **10-K** sections live in `sections/item-N-*.md`. MD&A is `item-7-mda.md`.
  Risk Factors is `item-1a-risk-factors.md`. Business overview is `item-1-business.md`.
- **10-Q** sections include Part I items (financial statements, MD&A) and
  Part II items (legal proceedings, risk factor updates). Slugs include the
  item number, e.g. `item-2-mda.md` (Part I MD&A) vs `item-1a-risk-factors.md`
  (Part II risk factors). Use the part context from the filename.
- **8-K** is event-driven: each filing covers specific items like 2.02
  (results of operations), 5.02 (officer departures), 7.01 (Reg FD), 9.01
  (financial statements & exhibits). Most 8-K substance is in `EX-99.1`
  exhibits (press releases). Always check `exhibits/` for 8-K filings.

## Exhibits — what's kept, what's not

Kept (substantive):
- `EX-3.x` — articles, bylaws (rare changes but material)
- `EX-10.x` — material contracts (credit, exec comp, supplier agreements)
- `EX-19` — insider trading policy (post-2024 SEC rule)
- `EX-21` — list of subsidiaries
- `EX-99.x` — press releases, financial supplements (highest-signal for 8-K)

Skipped (boilerplate or data-only):
- `EX-23` — auditor consents
- `EX-31` / `EX-32` — SOX certifications (boilerplate, identical across filers)
- `EX-101` / `EX-104` — XBRL / interactive data
- All non-text extensions (`.jpg`, `.xlsx`, `.zip`, etc.)

## Section parsing notes

Section parsing is regex-based on `Item N` markdown headings. When parsing
fails, `metadata.json` will show `"section_keys": []` — fall back to grep
over `filing.md`.
