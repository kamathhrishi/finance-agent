# `fs_research_agent` — filesystem-based financial research

A self-contained experiment that gives an LLM **filesystem tools only** (`ls`,
`read_file`, `grep`, `glob`) over a corpus of SEC filings stored as markdown,
and asks it to act as a financial research analyst.

The hypothesis: *for SEC research, raw filesystem access + a strong system
prompt beats a chunk-based vector-retrieval pipeline.* Validated by direct
side-by-side against the platform's existing `OrchestratorAgent` ("the deep
agent") on the same questions — and again by the FinanceBench benchmark
included in this folder.

This document is the long-form reference. Skim the table of contents and
jump to the section you need.

---

## Table of contents

1. [What this is and why it exists](#1-what-this-is-and-why-it-exists)
2. [Architecture at a glance](#2-architecture-at-a-glance)
3. [Quick start](#3-quick-start)
4. [Configuration / environment variables](#4-configuration--environment-variables)
5. [Module-by-module walkthrough](#5-module-by-module-walkthrough)
6. [Corpus layout and conventions](#6-corpus-layout-and-conventions)
7. [Plugging the agent into the chat backend](#7-plugging-the-agent-into-the-chat-backend)
8. [Citations and the document viewer](#8-citations-and-the-document-viewer)
9. [Tech-universe batch ingest](#9-tech-universe-batch-ingest)
10. [FinanceBench benchmark](#10-financebench-benchmark)
11. [Results so far](#11-results-so-far)
12. [Known issues and open work](#12-known-issues-and-open-work)
13. [Lessons learned](#13-lessons-learned)

---

## 1. What this is and why it exists

The platform's main RAG pipeline (`agent/rag/...`, `agent/orchestrator/...`)
uses semantic chunking + vector retrieval to fetch passages from SEC filings.
That pipeline works but has the typical chunk-RAG failure modes:

- The chunker decides what counts as a "passage", and gets it wrong sometimes.
- The retriever ranks by embedding similarity, which is OK at finding
  *relevant* passages but bad at finding *exact* lines (e.g. a specific row
  in a table).
- The agent never sees the broader context around a chunk — it gets the
  isolated chunk and has to synthesize.

`fs_research_agent` takes the opposite approach. The corpus is plain markdown
on local disk, organized into a self-describing folder layout. The agent has
four tools — `ls`, `read_file`, `grep`, `glob` — and no other capabilities.
It's expected to read a `README.md` and an `INDEX.md` to orient itself, then
navigate the filesystem like a human analyst would.

The model is **gpt-5.4-mini-2026-03-17** by default (started on nano, switched
to mini for higher TPM and better arithmetic). No LangChain, no embeddings,
no vector DB.

### Why this works

- **Zero retrieval lossiness.** The agent reads exactly the bytes that are in
  the source document. There is no chunk boundary to misalign with table
  rows, no embedding to lose precision, no reranker to second-guess.
- **Citations are unambiguous.** Every cite is a `path:line` pointer into a
  file the agent actually read. The document viewer can highlight that exact
  line range without fuzzy-matching.
- **Debugging is trivial.** Every tool call is captured in a per-question
  trace JSONL. You can replay exactly what the model saw and decided.
- **The corpus is human-readable.** You can `cd` into `data/filings/AAPL/`
  and open the files yourself.

### When this approach struggles

- Large corpora — `grep` over 50GB needs scoping to stay fast (we glob to
  ticker/form/year, never grep the whole tree).
- Computations the model has to do in its head — the agent reliably picks
  the right line items, then sometimes flubs basic arithmetic on the way to
  a final answer (`calc()` tool would fix this).
- Questions that require connecting facts across many filings — currently
  the agent does this serially within its tool budget; a more capable model
  would parallelize internally.

---

## 2. Architecture at a glance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ User question (chat UI, CLI, or benchmark runner)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FilesystemResearchAgent.run(question)                                       │
│   ReAct loop: LLM → tool calls → results → LLM → ... → final answer         │
│   - Driver:  gpt-5.4-mini-2026-03-17                                        │
│   - Budget:  25 tool calls per question (configurable)                      │
│   - Retries: 5x backoff, honors `retry-after` headers                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼  tool calls go through Sandbox
┌─────────────────────────────────────────────────────────────────────────────┐
│ tools.py: ls / read_file / grep / glob                                      │
│   All sandboxed under a single `data_root`. Path traversal is rejected.     │
│   `grep` shells out to ripgrep.                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼  reads from
┌─────────────────────────────────────────────────────────────────────────────┐
│ data/  (or benchmarks/financebench/data/, or any data_root you point at)    │
│                                                                             │
│   README.md             ← agent reads this first                            │
│   INDEX.md              ← top-level: ticker → form counts                   │
│   filings/<TICKER>/                                                         │
│     INDEX.md            ← per-ticker: every filing for that ticker          │
│     10-K/<FY>/                                                              │
│       filing.md, metadata.json, sections/<item>.md, exhibits/EX-<n>.md      │
│     10-Q/<FY>/<Q>/      ← same internal layout                              │
│     8-K/<YYYY-MM-DD>/   ← keyed by event filing date                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼  populated by
┌─────────────────────────────────────────────────────────────────────────────┐
│ ingest.py — datamule downloads + datamule HTML→markdown + table cleanup     │
│   batch_ingest.py — bulk-ingest the tech universe (188 tickers × 3 forms)   │
│   relabel_corpus.py — fix fiscal-year folder labels post-hoc                │
│   markdown_cleanup.py — collapse duplicated table columns                   │
│   backfill_table_cleanup.py — apply column dedupe to existing files         │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Two corpora live side by side**:

- `fs_research_agent/data/` — the **main** corpus, populated by the tech-batch
  ingest (188 Mega+Large IT/Comm tickers × 5 years × 10-K + 10-Q + 8-K +
  filtered substantive exhibits, ~12,000 filings, ~2.8 GB).
- `fs_research_agent/benchmarks/financebench/data/` — the **benchmark** corpus,
  isolated so the FinanceBench evaluation can't accidentally pull data from
  unrelated tech filings. Contains only the filings the FinanceBench questions
  reference.

---

## 3. Quick start

### Run a single question via CLI

```bash
python -m fs_research_agent.cli "What was NVDA's data center revenue in FY2024?"
```

The CLI prints the agent's reasoning trace to stderr and the streamed answer
to stdout. Useful flags:

- `-v` — verbose tool output (full result previews)
- `--raw` — emit raw event JSON, one per line
- `--budget 40` — bump the tool-call budget (default 25)
- `--model gpt-5.4-mini-2026-03-17` — override model

### Ingest a single ticker

```bash
DATAMULE_SEC_USER_AGENT="YourName your@email.com" \
  python -m fs_research_agent.ingest NVDA --years 5
```

This downloads the last 5 years of 10-K + 10-Q + 8-K filings (with
substantive exhibits), writes them under `fs_research_agent/data/filings/NVDA/`,
parses sections per Item, and regenerates the per-ticker `INDEX.md` plus
top-level `INDEX.md`.

### Batch-ingest the full tech universe

```bash
python -m fs_research_agent.batch_ingest --years 5
```

188 tickers × 3 forms = 564 jobs. Checkpointed at
`fs_research_agent/data/_batch_checkpoint.json` — re-running skips done jobs.
`--retry-failed-only` retries failures without redoing successes.

### Run the FinanceBench benchmark

```bash
# 1. Make sure required filings are in the benchmark corpus
python -m fs_research_agent.benchmarks.financebench.cli download

# 2. Run the agent against all 131 questions
python -m fs_research_agent.benchmarks.financebench.cli run --run-name run1

# 3. Inspect results
ls fs_research_agent/benchmarks/financebench/results/run1/
```

See [Section 10](#10-financebench-benchmark) for full benchmark details.

### Plug the agent into the chat backend

Set in `.env`:

```
USE_FS_RESEARCH_AGENT=true
```

Restart the FastAPI server. The chat router now uses
`FilesystemResearchOrchestrator` (in `orchestrator_adapter.py`) instead of
the legacy `OrchestratorAgent`. Citations render as `[10K-N]` / `[10Q-N]` /
`[8K-N]` markers and click through to the existing SEC filing viewer.

---

## 4. Configuration / environment variables

### System dependencies

- **`ripgrep`** (`rg`) must be on `PATH`. The agent's `grep` tool shells out to it (see `tools.py`); without it every grep call returns an error and the agent grinds to a halt. Install: `apt-get install ripgrep` or `brew install ripgrep`. On Railway/Nixpacks deploys this is provisioned by `nixpacks.toml` at the repo root (`aptPkgs = ["ripgrep"]`). Boot logs print `✅ ripgrep found at <path>` or a loud warning if missing.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `USE_FS_RESEARCH_AGENT` | `false` | When `true`, the chat backend uses this agent instead of `OrchestratorAgent`. Precedence: this > `USE_DEEP_AGENT` > `RAGAgent`. |
| `DATAMULE_SEC_USER_AGENT` | `John Smith johnsmith@gmail.com` | Set this to a real `Name email@domain` string. SEC throttles datamule's default UA — you'll get 500s on hot tickers (AAPL etc.) without a real one. |
| `FS_RESEARCH_DATA_ROOT` | `<pkg>/data/` | Override the corpus root used by the API and the agent's `Sandbox`. |
| `FS_RESEARCH_FINANCEBENCH_DATA_ROOT` | `<pkg>/benchmarks/financebench/data/` | Override the benchmark corpus location. |
| `FS_RESEARCH_TOOL_BUDGET` | `30` | Default per-question tool call budget when invoked through the orchestrator adapter. The chat router's `max_iterations` is intentionally ignored because it ships `3` for legacy ReAct agents (where 3 iterations = many parallel tool calls); the FS agent counts tool calls individually. |
| `OPENAI_API_KEY` | — | Required. |

---

## 5. Module-by-module walkthrough

```
fs_research_agent/
├── README.md                       ← you are here
├── __init__.py
├── agent.py                        ← the ReAct loop
├── tools.py                        ← ls / read_file / grep / glob + Sandbox
├── prompts.py                      ← system prompt
├── citations.py                    ← parse `path:line` cites from the answer
├── highlight.py                    ← line-range highlighting for the viewer
├── orchestrator_adapter.py         ← drop-in adapter for the chat router
├── cli.py                          ← debug CLI
├── ingest.py                       ← single-ticker ingest (datamule wrapper)
├── batch_ingest.py                 ← bulk universe ingest
├── universe.py                     ← read US_TECH_CLEANED.json → ticker list
├── relabel_corpus.py               ← fix existing wrong fiscal-year labels
├── markdown_cleanup.py             ← collapse duplicated table columns
├── backfill_table_cleanup.py       ← apply markdown_cleanup to existing files
├── data/                           ← main corpus (gitignored)
└── benchmarks/
    ├── __init__.py
    └── financebench/
        ├── __init__.py
        ├── dataset.py              ← load HF FinanceBench → questions
        ├── judge.py                ← LLM-as-judge (1-10 scoring)
        ├── download.py             ← ensure required filings in benchmark corpus
        ├── runner.py               ← drive agent through every question
        ├── cli.py                  ← `download` / `run` / `summary` subcommands
        ├── data/                   ← isolated benchmark corpus (gitignored)
        └── results/<run-name>/
            ├── results.jsonl       ← one JSON per question
            ├── summary.json        ← aggregate stats
            └── traces/<id>.jsonl   ← per-question full event trace
```

### `agent.py` — `FilesystemResearchAgent`

The ReAct loop. `run(question)` is an `async` generator yielding event dicts:

- `progress` — orientation messages
- `llm_call` — each LLM call, with phase + tool counts
- `tool_start` / `tool_end` — every tool invocation; `tool_end` carries
  `result_chars`, `result_preview` (300 chars), and `result_full` (the entire
  result — for trace capture)
- `token` — streamed final answer, word at a time
- `result` — final answer + counts
- `error` — agent failure

OpenAI client uses `temperature=0`. `reasoning_effort` is intentionally NOT
set — gpt-5.4 family rejects it when tools are present on
`/v1/chat/completions`. (You'd need `/v1/responses` to use reasoning_effort
with tools, which is a bigger refactor.)

**Rate-limit handling** (`_wait_from_429`):
1. Honor `retry-after` HTTP header.
2. Honor `Retry-After-Ms` HTTP header.
3. Parse "try again in 235ms" / "in 1.2s" out of the error body (OpenAI's
   TPM-cap message includes this when it knows exactly when budget refills).
4. Fallback to exponential backoff starting at 5s (NOT 1s — TPM windows are
   seconds-scale).

Clamps to `[1.0, 60.0]` seconds.

### `tools.py` — sandboxed filesystem tools

All four tools route through `Sandbox.resolve(user_path)`, which:

1. Resolves the user-supplied path relative to the sandbox root.
2. Calls `os.path.commonpath([candidate, root])` and rejects anything that
   doesn't share the root prefix. Path traversal (`../../../etc/passwd`)
   raises `SandboxViolation`.

Tools cap their output to keep tool results bounded:

- `read_file` — paginated, default 400 lines, max 1500 per call. Output
  prefixed with line numbers.
- `grep` — caps at 12,000 chars total response, 200 matches/file, 5 lines of
  context. Shells out to `rg` (ripgrep is a hard dependency).
- `ls` — truncates the listing only with a footer counter, never silently.
- `glob` — caps at 200 matches.

Then the agent itself caps tool results at 16,000 chars before they reach
the LLM.

### `prompts.py` — system prompt

Major sections:

- **Discovering the corpus** — read README, then INDEX, then per-ticker INDEX.
  Treats INDEX.md as the source of truth ("if INDEX.md lists `<TICKER>`, that
  ticker IS in the corpus — don't `ls filings` to double-check").
- **Layout you will find** — describes the ticker/form/period folder schema
  and the per-form quirks (10-K is annual, 10-Q has fiscal Q labels that may
  not match calendar Q for non-Dec fiscal years, 8-K is event-driven and
  often has the press release in `EX-99.1`).
- **Citation format** — strict `(filings/<TICKER>/<FORM>/<FY>/sections/<item>.md:<line>)`.
  Forbids numbered shorthand `(1)`, `[3]` and abbreviated paths
  `(.../FY####/...:NN)`. Repetition is encouraged ("write the full path five
  times if needed").
- **Period locking** — for any period-specific question, locate the folder,
  read `metadata.json` to verify `period_of_report`, lock to the right
  column when reading multi-year tables.
- **Standard analyst formulas** — table of definitions (operating margin,
  quick ratio, FCF conversion, ROA, etc.). The agent uses these when the
  question doesn't provide a formula.
- **Specific drivers** — for "what drove X" questions, name the specific
  litigation / acquisition / charge from the filing instead of generic
  categories ("macro headwinds").
- **Self-check** — verify period match, formula correctness, driver
  specificity before writing the final answer.
- **Writing the answer** — imperative voice ("ELABORATE and COMPREHENSIVE
  responses with MAXIMUM DETAIL"), bold all financial figures, use markdown
  tables for ≥2 values across periods/segments. Lifted from the platform's
  existing `agent/rag/response_generator.py` because nano-class models
  default to terse list-mode without imperative pressure.

### `citations.py` — parse `path:line` → `[10K-N]`

The agent emits citations like
`(filings/NVDA/10-K/FY2025/sections/item-7-mda.md:42-58)`.
`extract_citations(answer, corpus_root)`:

1. Walks the answer with a regex matching the canonical path:line[-line]
   format.
2. For each unique cite, generates a marker (`[10K-N]`, `[10Q-N]`, `[8K-N]`
   depending on the form parsed from the path).
3. Replaces inline cites in the answer with `[FORM-N]` markers.
4. Builds a list of structured `FsCitation` objects with the data the
   frontend needs (ticker, fiscal_year, section human label, line range,
   `chunk_text` snippet for the citation card).
5. Strips abbreviated forms `(.../FY####/...:NN)` as a defensive cleanup.

Marker prefix matters: the frontend's `ChatMessage.preprocessCitationMarkers`
regex matches `10K-N`, `10Q-N`, `8K-N` (and a few legacy patterns) — anything
else renders as plain text. The 8K branch was added during this work.

### `highlight.py` — line-range mark injection

`inject_line_highlights(markdown, [LineHighlight, ...])` wraps each cited
line range in a `<mark class="highlighted-chunk" data-chunk-id="...">` span.
Heading prefixes (`##`) are preserved before the open tag so markdown still
parses correctly.

Used by `app/routers/fs_research.py`'s `/fs-research/document/with-highlights`
endpoint.

### `orchestrator_adapter.py` — `FilesystemResearchOrchestrator`

Drop-in agent compatible with `app/routers/chat.py`. Implements the same
`execute_rag_flow()` async generator contract as `OrchestratorAgent` and
`RAGAgent`, so the chat router doesn't know which agent is behind it.

Key behaviors:

- Maps the FS agent's events to the chat router's event taxonomy
  (`reasoning` / `progress` / `token` / `result`).
- **Humanizes tool messages**: `read_file('filings/NVDA/.../item-7-mda.md')`
  becomes `"Reading NVDA's MD&A (FY2025)…"`. Section name dictionary in
  this file (short labels for the live trace; longer labels live in
  `citations.py`).
- **Suppresses noisy events**: drops `llm_call` events entirely, drops
  `tool_end` summaries except when grep returns zero matches (informative,
  explains why agent retries).
- **Final-answer rewrite**: streams the answer AFTER citation rewriting
  so users see `[10K-N]` markers, not raw `path:line` strings.
- **Ignores `max_iterations` from the chat router** — it ships `3` by
  default (a legacy ReAct iteration count), which would strangle the FS
  agent. Uses its own `max_tool_calls` budget instead.

### `ingest.py` — single-ticker ingest

Wraps `datamule.Portfolio.download_submissions()` + per-form section
parsing + exhibit filtering + metadata writing.

Form parsers (regex-based on markdown headings, since datamule emits
`## Item 1. Business`):

- `_split_10k` — uses last-occurrence rule to skip TOC mentions.
- `_split_10q` — splits Part I from Part II first via `## PART I` / `## PART II`
  headings, then runs item parsers within each region. Items are keyed
  `p1-1`, `p1-2`, `p2-1`, etc.
- `_split_8k` — flat decimal items (Item 2.02, 5.02, 7.01, 9.01, etc.).

Exhibit filter (`_is_keepable_exhibit`):
- **Keep**: `EX-3.x` (articles), `EX-10.x` (material contracts), `EX-19`
  (insider trading policy), `EX-21` (subsidiaries), `EX-99.x` (press
  releases / financial supplements).
- **Skip**: `EX-23` (auditor consents), `EX-31`/`EX-32` (SOX certs — boilerplate),
  `EX-101`/`EX-104` (XBRL data).
- Skip non-text extensions (`.jpg`, `.xlsx`, `.zip`, etc.).

Period resolution:
- Reads `meta["period"]` (datamule's name for SEC's `<PERIOD>` SGML field,
  format `YYYYMMDD`, e.g. `20240928` for Apple's FY2024 10-K).
- Falls back to `period-of-report` / `period_of_report` if missing.
- Final fallback to `filing_date` (worst case — see [Section 12](#12-known-issues-and-open-work)).

This was THE biggest bug in the corpus. Earlier code used `filing_date` only,
which labeled a 10-K filed Feb 2018 (covering fiscal 2017) as `FY2018`. Every
filing in the original 12,206-doc corpus was off by one. Fixed in `ingest.py`
and backfilled in-place via `relabel_corpus.py`.

Retry-with-backoff for SEC EDGAR transient 500s (4 attempts, exponential).

### `batch_ingest.py` — universe-scale ingest

Iterates the tech universe (188 tickers × 3 forms = 564 jobs).
Per-(ticker, form) checkpoint at `data/_batch_checkpoint.json`. Re-running
skips done jobs by default; `--retry-failed-only` retries failures only;
`--no-skip` re-ingests everything from scratch.

### `universe.py` — load tech-universe ticker list

Reads `agent/rag/data_ingestion/US_TECH_CLEANED.json` (the platform's
existing curated list of 2,438 US tech companies from FinanceDatabase).
Default filter: Mega+Large Cap × Information Technology + Communication
Services → 188 tickers. Override sectors / market caps via function args.

### `relabel_corpus.py` — fix existing folders

When the period-of-report fix landed, we had 12K+ filings already on disk
under wrong fiscal-year labels. This script:

1. For each ticker, fetches SEC's submissions JSON
   (`https://data.sec.gov/submissions/CIK<padded>.json`) — one HTTP request
   per ticker, returns `accessionNumber` + `reportDate` for every filing.
2. Matches each existing filing folder to its accession number → looks up
   the correct `period_of_report`.
3. Renames folders via two-pass staging (move to
   `<form_root>/.stage-<accession>/` first, then to final destination) so
   swapped destinations don't collide.
4. Updates each filing's `metadata.json` with corrected period_of_report
   and recomputed labels.
5. Regenerates per-ticker INDEX and top-level INDEX at the end.

Idempotent. Run with `--dry-run` to preview moves.

### `markdown_cleanup.py` — collapse SEC's duplicated table columns

The big revelation from the FinanceBench investigation. Datamule preserves
SEC's HTML structure where every dollar value occupies TWO adjacent columns
(one for `$` symbol, one for the value). Result:

```
| Operating income | 1,493,602 | 1,493,602 | 903,095 | 903,095 | 412,685 | 412,685 |
```

Each fiscal year is duplicated. The model reads this and grabs the wrong
column pair. Adobe's recurring "FY2015 → FY2016 op income YoY" failure
(118.8% vs gold 65.4%) was 100% caused by this — agent picked column 3 + 5
thinking they were FY16 + FY15, but they were the duplicates of FY15 + FY14.

`cleanup_markdown_tables(markdown)` walks every table in the document and
collapses adjacent-duplicate column pairs to a single column. Handles three
sub-patterns:

- `| 1,235 | 1,235 |` — value duplicated.
- `|  | $1,235 |` — left empty, right has $-prefixed value.
- `| (1,508 | (1,508) |` — datamule splits negative-number parens across cells.

Conservative threshold: 80% of data rows must match one of these patterns
before the columns collapse.

Hooked into the ingest pipeline (`_doc_markdown` calls
`cleanup_markdown_tables` before saving).

### `backfill_table_cleanup.py` — apply cleanup to existing files

Walks the corpus and runs `cleanup_markdown_tables` on every `filing.md`,
`sections/*.md`, and `exhibits/*.md`. Idempotent. Saves ~15% on cleaned
files (from removing the redundant columns).

---

## 6. Corpus layout and conventions

```
data/                                     ← FS_RESEARCH_DATA_ROOT
├── README.md                             ← agent reads this first
├── INDEX.md                              ← top-level: ticker → form counts
└── filings/
    └── <TICKER>/                         ← e.g. NVDA, MSFT, MMM
        ├── INDEX.md                      ← per-ticker: every filing for this ticker
        ├── 10-K/
        │   └── <FY-LABEL>/               ← FY####, e.g. FY2025
        │       ├── filing.md             ← full filing text in markdown
        │       ├── metadata.json         ← ticker, cik, period_of_report, etc.
        │       ├── sections/             ← parsed item files (when parsing succeeds)
        │       │   ├── item-1-business.md
        │       │   ├── item-1a-risk-factors.md
        │       │   ├── item-7-mda.md
        │       │   ├── item-7a-quant-qual-disclosures.md
        │       │   ├── item-8-financial-statements.md
        │       │   └── ...
        │       └── exhibits/             ← filtered substantive exhibits
        │           ├── EX-10.1.md        ← material contracts
        │           ├── EX-19.md          ← insider trading policy
        │           ├── EX-21.md          ← subsidiaries
        │           └── EX-99.1.md        ← press releases
        ├── 10-Q/
        │   └── <FY-LABEL>/
        │       └── <QUARTER>/            ← Q1 / Q2 / Q3 / Q4
        │           └── (same internal layout)
        └── 8-K/
            └── <YYYY-MM-DD>/             ← keyed by filing date (events)
                └── (same internal layout)
```

### Conventions worth knowing

- **`<FY-LABEL>` is `FY<YYYY>` from the period of report**, NOT from the
  filing date. A 10-K filed Feb 2018 covering fiscal year ending Dec 31 2017
  lives in `FY2017/`. Apple (fiscal year ends late September) — a 10-K
  filed Nov 2024 covering period ending Sep 28 2024 lives in `FY2024/`.

- **`<QUARTER>` is calendar quarter of the period-end month**, NOT fiscal
  quarter. For a calendar-year company (Dec fiscal year-end) these align.
  For non-Dec fiscal years (NVDA fiscal year ends late January, MSFT in
  late June), the calendar Q labels in our folder names won't match the
  company's own fiscal Q labeling. The agent is told this in the README and
  the system prompt — it's expected to confirm via `metadata.json`.

- **`metadata.json`** carries the ground truth:
  ```json
  {
    "ticker": "NVDA",
    "cik": "0001045810",
    "form": "10-K",
    "fiscal_year_label": "FY2025",
    "quarter_label": null,
    "filing_date": "2025-02-25",
    "period_of_report": "2025-01-26",
    "accession": "000104581025000023",
    "filing_chars": 581234,
    "section_keys": ["1", "1A", "1C", "2", "3", "5", "7", "7A", "8", "9A", "9B"],
    "exhibits": [
      {"type": "EX-21", "file": "EX-21.md", "chars": 5648},
      {"type": "EX-19", "file": "EX-19.md", "chars": 8912}
    ]
  }
  ```

- **`section_keys: []`** means item parsing failed for that filing — fall
  back to grepping `filing.md` directly.

- **Top-level `INDEX.md`** carries a `Ticker | Company | 10-K | 10-Q | 8-K`
  table. The Company column is critical — without it, the agent sees `MMM`
  in the table and doesn't connect it to a question asking about "3M".
  (Discovered when 3M questions started failing on the larger benchmark
  corpus.)

---

## 7. Plugging the agent into the chat backend

Set `USE_FS_RESEARCH_AGENT=true` in `.env` and restart `uvicorn`. The chat
router will load `FilesystemResearchOrchestrator` instead of
`OrchestratorAgent`.

The agent factory in `agent/__init__.py` selects in this precedence order:

1. `USE_FS_RESEARCH_AGENT=true` → `FilesystemResearchOrchestrator`
2. `USE_DEEP_AGENT=true` (default) → `OrchestratorAgent` (legacy deep agent)
3. otherwise → `RAGAgent` (legacy chunk-RAG)

The orchestrator adapter implements:
- `execute_rag_flow(question, ...)` async generator — primary entry point
- `execute_rag_flow_async(question, **kwargs)` — collects stream into a single result
- `set_database_connection(db)` — no-op (FS agent doesn't touch DB)
- `set_user_context(**kwargs)` — no-op

Final `result` event shape (matches the existing chat router contract):

```python
{
  "type": "result",
  "data": {
    "response": {
      "answer": "<markdown answer with [10K-N] inline markers>",
      "citations": [
        {
          "marker": "[10K-3]",
          "chunk_id": "10K-3",
          "path": "filings/NVDA/10-K/FY2025/sections/item-7-mda.md",
          "line_start": 42,
          "line_end": 58,
          "chunk_text": "Data center revenue grew to ...",
          "ticker": "NVDA",
          "filing_type": "10-K",
          "fiscal_year": 2025,
          "section": "MD&A (Item 7)",
          "type": "10-K",
          "citation_type": "10-K",
          "source_backend": "fs_research",
          "relevance_score": 0.9
        }
      ]
    },
    "timing": {"total_seconds": 12.4, "agent_seconds": 11.8},
    "stats": {"tool_calls": 14, "llm_calls": 5, "citation_count": 7},
    "backend": "fs_research"
  }
}
```

---

## 8. Citations and the document viewer

Citations from the FS agent ride the same `ChatCitation` schema as the
existing platform but with three extra fields:

- `path` — corpus-relative path to the file
- `line_start`, `line_end` — 1-based line range (inclusive)
- `source_backend = "fs_research"` — routes the click handler

The frontend `ChatMessage.handleViewSECFiling` notices
`source_backend === "fs_research"` and routes the click to the new
`SECFilingViewer` mode, which calls
`POST /fs-research/document/with-highlights` (in `app/routers/fs_research.py`)
with `{path, relevant_chunks: [{chunk_id, line_start, line_end, primary}]}`.
The endpoint reads the file from the FS corpus, wraps the cited line range
in `<mark data-chunk-id="...">`, returns markdown. The viewer renders with
ReactMarkdown + rehypeRaw and scrolls to the primary mark.

Marker format: `[10K-N]` / `[10Q-N]` / `[8K-N]`. The frontend's
`preprocessCitationMarkers` regex matches all three (the 8K branch was added
specifically for this work).

---

## 9. Tech-universe batch ingest

### What gets downloaded

`universe.load_universe()` reads
`agent/rag/data_ingestion/US_TECH_CLEANED.json` and applies a default filter:

- Sectors: Information Technology + Communication Services
- Market cap: Mega Cap + Large Cap

Result: **188 tickers** (down from 2,438 raw).

`batch_ingest.run_batch()` then iterates each ticker × form, downloading the
last `years` years of filings via `ingest_form_for_ticker`. Default 5 years.

### Checkpoint

`data/_batch_checkpoint.json` tracks each `(ticker, form)` job:

```json
{
  "MSFT|10-K": {
    "ok": true,
    "filings_written": 5,
    "error": null,
    "elapsed_s": 18.4,
    "ts": 1735761023,
    "years": 5,
    "keep_exhibits": true
  },
  "AAPL|10-K": {
    "ok": false,
    "error": "ClientResponseError: 500, message='Internal Server Error', ...",
    ...
  }
}
```

Re-running `batch_ingest` skips `ok: true` jobs by default. Useful flags:

- `--retry-failed-only` — only retry jobs where `ok: false`
- `--no-skip` — re-ingest everything
- `--tickers MSFT,AAPL` — restrict to specific tickers (overrides universe)
- `--forms 10-K,10-Q` — restrict to specific forms

### Common gotchas

- **SEC EDGAR's User-Agent throttle.** Datamule's default UA
  (`John Smith johnsmith@gmail.com` — yes, literally) gets rate-limited /
  500'd by SEC for hot tickers like AAPL. Set
  `DATAMULE_SEC_USER_AGENT="StrataLens kamathhrishi@gmail.com"` (or similar)
  before running.
- **SEC EDGAR's transient 500s.** Even with a real UA, the EFTS search
  endpoint returns 500 occasionally on rapid back-to-back queries.
  `_download_form_to_scratch` has 4-attempt exponential backoff retry built in.
- **Delisted tickers.** SQ (Block — renamed to XYZ in 2024) and ATVI
  (Activision Blizzard — acquired by Microsoft Oct 2023) no longer appear
  in SEC's `company_tickers.json`. Datamule fails to look up their CIK.
  Workaround: hardcode the historical CIK and call datamule with `cik=...`
  instead of `ticker=...`. Currently `dataset.py`'s `COMPANY_TO_TICKER` map
  excludes these two from the FinanceBench benchmark to avoid silent
  missing-data failures.

---

## 10. FinanceBench benchmark

### Overview

[FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) is a
150-question dataset from PatronusAI testing financial analyst tasks against
SEC filings. Each question carries:

- `company` (display name, e.g. "3M")
- `doc_name` (specific filing it references, e.g. `3M_2018_10K`)
- `doc_type` (10K / 10Q / 8K / Earnings)
- `doc_period` (year)
- `question`
- `answer` (gold)
- `justification` (gold reasoning)
- `evidence` (text snippets)

Of the 150 questions, **131 are runnable** here:

- 14 earnings-call transcript questions are skipped (no transcripts in
  corpus yet).
- 5 SQ + ATVI questions are skipped (delisted tickers).

### Components

- **`dataset.py`** — `load_questions()` returns a list of
  `FinanceBenchQuestion` objects with parsed (form, year, quarter | filing_date).
  `required_filings(qs)` returns the deduped set of filings needed to answer
  every question.

- **`download.py`** — `ensure_required_filings()` calls the existing ingest
  pipeline for every (ticker, form) pair the questions reference.
  Writes into the **isolated** `BENCHMARK_DATA_ROOT`
  (`fs_research_agent/benchmarks/financebench/data/`) — never the main
  corpus.

- **`runner.py`** — `run_benchmark()` instantiates the agent sandboxed to
  the benchmark corpus, loops through questions with bounded
  `asyncio.Semaphore` concurrency, judges each answer, writes
  `results.jsonl` + `summary.json` + per-question `traces/<id>.jsonl`.

- **`judge.py`** — single-shot LLM judge. Score 1-10. ≥7 counts as correct.
  Verbose-tolerant rubric: "a verbose, well-cited answer that contains the
  correct core fact MUST score ≥7. Length is never a reason to downscore."

- **`cli.py`** — three subcommands:
  ```bash
  python -m fs_research_agent.benchmarks.financebench.cli download [--companies X,Y,Z] [--dry-run]
  python -m fs_research_agent.benchmarks.financebench.cli run --run-name nano_full [--limit 5] [--companies X,Y]
  python -m fs_research_agent.benchmarks.financebench.cli summary <run-name>
  ```

### Question augmentation (the "period hint")

FinanceBench questions implicitly assume a specific filing context (`doc_name`)
that's NOT in the question text. Without that hint the agent guesses the
latest 10-K, which is wrong for half the questions. The runner appends a
one-line hint:

```
<question>

_(For grounding: this question refers to **<Company>**'s
<TICKER> 10-K for fiscal year <YEAR>. Use the corresponding folder in
the corpus.)_
```

This is the minimum information needed to disambiguate filing year without
giving away the answer. Lifted accuracy ~5pp in benchmark testing.

### Per-question results

`results.jsonl` — one JSON per question:

```json
{
  "id": "financebench_id_03029",
  "company": "3M",
  "ticker": "MMM",
  "form": "10-K",
  "year": 2018,
  "question": "...",
  "expected": "$1577.00",
  "predicted": "FY2018 capital expenditures... was **$1,577 million** (`...md:216`).",
  "justification": "Line item: Purchases of PP&E ...",
  "judge_score": 10,
  "judge_reasoning": "Predicted matches expected exactly...",
  "judge_passed": true,
  "tool_calls": 8,
  "llm_calls": 3,
  "elapsed_s": 16.0,
  "error": null
}
```

`summary.json` — aggregate stats including `by_form` and `by_company` splits.

`traces/<id>.jsonl` — every event the agent emitted for that question,
including full tool results. Used for forensic analysis of failures.

### Resume

Re-running with the same `--run-name` skips questions already in
`results.jsonl`. Use `--no-resume` to re-run everything.

---

## 11. Results so far

Each row is the same 21 questions (or 131 for full runs) on the FinanceBench
benchmark. "Score" is the LLM judge's 1-10 mean.

### Smoke runs (3M / AMD / Adobe — 21 questions)

| Run | Pass | Avg | What changed |
|---|---|---|---|
| v1 (`smoke_3co`) | 5/18 (28%) | 4.4 | initial — buggy FY labeling |
| v2 (`smoke_3co_v2`) | 15/21 (71%) | 7.95 | fixed period-of-report labeling |
| v3 (`smoke_3co_v3`) | **17/21 (81%)** | 8.62 | + year-locking, standard formulas, specific drivers, self-check |
| v4 (`smoke_3co_v4`) | 16/21 (76%) | 8.00 | + "show your work" + diagnostic-question rules — REGRESSED on simple lookups (over-engineered) |
| v5 (`smoke_3co_v5`/`v5b`) | 13-14/21 (62-67%) | 7.05-7.33 | reverted v4 prompt; larger benchmark corpus broke 3M lookup ("MMM not in corpus" hallucination) |
| v6 (`smoke_3co_v6`) | 14/21 (67%) | 7.33 | INDEX-is-truth fix (don't re-verify with `ls`) — partial |
| v7 (`smoke_3co_v7`) | 16/21 (76%) | 8.33 | + Company-Name column in INDEX.md (so agent sees `MMM = 3M`) |

### Full benchmark runs (131 questions)

| Run | Pass | Avg | What changed |
|---|---|---|---|
| `full_v7` | 75/136 (55%) | 6.65 | full benchmark, original (still some bugs) |
| `full_v8` | killed | — | better dedupe in flight |
| `full_v9` | killed | — | rate-limit storms (gpt-5.4-nano on shared TPM) |
| **`full_v10`** | **93/131 (71%)** | **7.63** | gpt-5.4-mini + better dedupe (mixed-pattern column collapse) + period hint + 5 ticker-not-found cases excluded |

**v10 lenient score (numerically-within-±5% counts as correct): 98/131 = 75%.**

### v10 by form

| Form | Pass |
|---|---|
| 10-K (107 q) | 80/107 = 74% |
| 10-Q (15 q) | 6/15 = 40% (still weak — quarter mapping) |
| 8-K (9 q) | 7/9 = 77% |

### v10 failure distribution (38 fails)

| Mode | Count |
|---|---|
| Wrong number from computation (right inputs!) | 15 |
| Mixed / other | 10 |
| Missed specific item in the right section | 4 |
| Wrong scope (e.g., total vs domestic stores) | 4 |
| Wrong directional conclusion with right data | 3 |
| Bailout / gave up | 2 |

The headline finding: **most failures aren't "agent picked the wrong number"
anymore** (the column dedupe and period-hint fixed those). They're "agent
picked the right number but flubbed the arithmetic on the way to a final
answer". A `calc(expr)` tool would fix ~5pp of the remaining gap.

---

## 12. Known issues and open work

### Solved (don't re-litigate)

- **FY labeling from filing-date.** Fixed in `ingest.py` (`_normalize_date`
  + `meta.get("period")` fallback chain). All existing filings relabeled
  via `relabel_corpus.py`.
- **Datamule's duplicated table columns.** Fixed in `markdown_cleanup.py`.
  Apply at ingest time; backfill existing files via `backfill_table_cleanup.py`.
- **Agent hallucinating "ticker not in corpus".** Fixed via INDEX-is-truth
  prompt rule + Company column in `INDEX.md`.
- **OpenAI rate-limit storms.** `_wait_from_429` now honors `retry-after`
  headers and parses "try again in Xms" hints. Switching judge to a
  separate model (or using gpt-5.4-mini end-to-end on its higher-TPM
  bucket) helps further.
- **gpt-5.4-nano arithmetic flubs.** Switched to gpt-5.4-mini default.
  Doesn't solve it entirely, but helps.

### Open

1. **`calc(expr)` tool.** ~30 lines. Would deterministically compute
   formulas. Estimated +5pp on FinanceBench (most arithmetic-error
   failures vanish).

2. **10-Q quarter mapping.** Currently uses calendar quarter from
   period-end month. For non-Dec fiscal years (J&J FY ends Dec, but
   PepsiCo's 13-week quarters end at oddball dates; MSFT FY ends late
   June), the agent needs to manually translate calendar Q to fiscal Q
   from `metadata.json`. Could compute a `fiscal_quarter_label` field at
   ingest time using `fiscal-year-end` from datamule's filer metadata.

3. **Renamed/delisted ticker support.** SQ (Block→XYZ), ATVI (acquired by
   MSFT). Datamule fails the ticker→CIK lookup. Solution: maintain a
   `<TICKER, historical_cik>` map and pass `cik=` directly.

4. **Section-routing for "what acquisitions / restructurings"** questions.
   Agent currently goes to MD&A; the named events often live in Notes to
   Financial Statements (Item 8). Could add a prompt-rule for these
   question shapes, or a section-aware router.

5. **Diagnostic question handling** ("is X capital-intensive?"). Agent
   sometimes computes the right ratios then gives the opposite directional
   verdict from the gold. Real synthesis weakness; a stronger model would
   help.

6. **Stronger model on synthesis only.** Current architecture uses one
   model end-to-end. Could keep gpt-5.4-mini for the research loop and use
   a stronger model (Opus, GPT-5 full) for the final synthesis step.
   Estimated +5-10pp on hard questions.

7. **Earnings-call transcripts.** FinanceBench has 14 transcript questions
   we currently skip. Need a transcript ingest pipeline.

8. **Backfill remaining incomplete dedupes.** First-pass `markdown_cleanup`
   was too conservative (only collapsed when ≥70% of rows had identical
   adjacent values). Improved version (handles mixed-pattern tables like
   BBY) is in place but the main corpus needs another backfill pass.

9. **Web/API exposure of the benchmark.** Currently the benchmark is
   CLI-only. A small endpoint that triggers a run + streams results to
   the frontend would make iteration faster.

---

## 13. Lessons learned

In rough order of "non-obvious things that turned out to matter":

1. **Data quality dwarfs model quality for retrieval-bound tasks.**
   The single biggest jump was 28% → 71% from fixing the period-of-report
   labeling bug. The model didn't change. The folder names did. Most
   subsequent gains came from fixing the markdown table format, not from
   prompt engineering.

2. **Trust the index, not the listing.** When the agent had to verify "is
   X in the corpus?" by `ls`-ing 32 directories, it often hallucinated
   that X wasn't there. When we told it "if INDEX.md lists X, X exists",
   the bailouts vanished. Tells you something about how nano-class models
   handle list-presence checks.

3. **Imperative voice matters more than rules.** v3's prompt got 81% with
   "ALWAYS MENTION ALL FINANCIAL FIGURES" + "Never omit a relevant
   figure". v4 added more structure ("show your work" block, diagnostic
   rules) and dropped 5pp because the agent over-engineered simple
   lookups. Less is more for nano/mini.

4. **The agent reads the work, not just the headline.** When asked
   "what drove margin?", agent reads the MD&A bridge table (which
   correctly shows "litigation: +1.4 pp") but then writes "raw material
   headwinds" instead of the specific litigation name from the surrounding
   paragraph. The bridge table tells you the magnitudes; the narrative
   tells you the named events. Both are required for a good answer.

5. **Pass the implicit context.** FinanceBench questions implicitly reference
   a specific filing year. Without that hint, the agent guesses the latest
   10-K and is 50/50 right. With a one-sentence hint
   ("this question refers to <Company>'s 10-K for FY<year>"), it's
   reliable.

6. **Judge calibration matters but is fixable.** Initial judge gave
   "PP&E" ≠ "PPNE" (treated as different) and "1.43%" ≠ "0.01"
   (different format, same number). The verbose-tolerant rubric helps
   but a numerical-equivalence post-pass catches the format issues.

7. **Per-question traces are non-negotiable for debugging.** The trace
   captures every tool call + full result. Without it, post-mortem on
   why a question failed is guesswork. With it, you can replay exactly
   what the model saw and decided in 30 seconds.

8. **Datamule is great but raw.** It solves the SEC EDGAR + HTML→markdown
   problem nicely, but its outputs need post-processing (column dedupe,
   period extraction) before they're agent-ready. Treat datamule as the
   wire format, not the final format.

9. **Sandboxing is cheap and worth it.** `os.path.commonpath` rejection
   is 5 lines and prevents the agent from wandering out of the corpus.
   Without it, every tool call is a path-traversal opportunity.

10. **The bimodal score distribution is the signature.** 60+ perfect 10s
    and 30+ score-3s, with very few in the middle. Tells you the agent
    either nails it or whiffs entirely — the failure modes are
    structural (wrong section, wrong year, wrong formula), not gradients.
    Fix the structure and a whole bucket of failures collapses at once.

---

*Last meaningful update: after `full_v10` run on 2026-05-01.
Current state: 71% strict / 75% lenient on FinanceBench (131 runnable Qs).*
