# StrataLens AI

Equity-research platform that answers questions about public-company SEC filings (10-K · 10-Q · 8-K). Built around a single research agent that reads filings directly off a local filesystem corpus — no chunking, no vector DB, no semantic routing.

**Live platform:** [www.stratalens.ai](https://www.stratalens.ai)

**Coverage:** 300+ US tech companies — semis, software, fintech, e-commerce, exchanges, EVs — with 3 years of 10-K, 10-Q, and 8-K history. Updated automatically by an in-process watcher polling SEC EDGAR.

---

## Architecture

```
                                  CHAT REQUEST
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │ FilesystemResearchAgent │
                          │     (gpt-5.4-mini)      │
                          └────────────┬────────────┘
                                       │ ReAct loop — picks one tool per turn
        ┌──────────────┬───────────────┼───────────────┬──────────────────┐
        ▼              ▼               ▼               ▼                  ▼
   ┌─────────┐   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
   │ ls/glob │   │ read_file│    │   grep   │    │  (more   │    │ news_search  │
   │         │   │          │    │ (ripgrep)│    │   FS)    │    │   (Tavily)   │
   └────┬────┘   └────┬─────┘    └────┬─────┘    └────┬─────┘    └──────┬───────┘
        │             │               │               │                 │
        └─────────────┴───────┬───────┴───────────────┘                 │
                              ▼                                         ▼
       ┌───────────────────────────────────────────┐    ┌───────────────────────────┐
       │   Local SEC corpus (markdown filesystem)  │    │     Tavily web API        │
       │   filings/<TICKER>/{10-K,10-Q,8-K}/...    │    │  Recent news + context    │
       │   Railway persistent volume.              │    │  for events POST a        │
       │   Hydrated from S3 on cold start.         │    │  filing's date.           │
       │   PRIMARY source — every cited number     │    │  OPTIONAL: silently       │
       │   resolves back here.                     │    │  skipped when             │
       │                                           │    │  TAVILY_API_KEY unset.    │
       └───────────────────────────────────────────┘    └───────────────────────────┘

       + Background watcher: polls SEC every 30 min, writes new filings to the volume
```

### Why the rebuild

This is the second incarnation of the agent. The first was a multi-agent / chunk-RAG pipeline — semantic routing → vector retrieval → cross-encoder rerank → iterative self-improvement. It was decent on benchmarks. Lots of people reached out about the approach. The original write-up: [Building a 10-K research agent with chunked RAG](https://substack.com/home/post/p-181608263).

But it was built for a different generation of models. Two things changed:

1. **Small models can use a terminal well.** A model that can read a directory listing, follow a markdown index, and grep for a number is doing the same thing an analyst does. No retrieval pipeline needed — just give it the filesystem.
2. **Frontier models know SEC filings.** They know what Item 7 is, what `MD&A` means, where to look for segment revenue. The agent doesn't need to be taught the corpus structure — it asks the corpus directly via `ls`.

So the rebuild deleted the vector DB, the chunk store, the rerank step, the question-classifier, and the iterative self-improvement loop. What's left is four filesystem tools (`ls`, `read_file`, `grep`, `glob`) over a sandboxed filesystem of pre-cleaned SEC markdown, plus an optional `news_search` tool that hits Tavily for recent context. The LLM does its own retrieval the same way an analyst would: list the directory, read the index, grep for the figure, quote the line.

Glad to be past solving retrieval — frees up energy for higher-value problems. Design notes in [`fs_research_agent/README.md`](fs_research_agent/README.md).

---

## SEC ingestion (datamule)

The corpus is downloaded from SEC EDGAR via [datamule](https://github.com/john-friedman/datamule-python) — a Python SDK around EDGAR's submissions API plus an HTML→markdown converter. We use one class (`Portfolio`) at two call sites:

- **Bulk seed** (`fs_research_agent/ingest.py`) — pulls every 10-K / 10-Q / 8-K for a ticker over the last N years. Used for the initial corpus build (`python -m fs_research_agent.batch_ingest`).
- **Watcher single-fetch** (`fs_research_agent/watcher.py`) — pulls one specific filing by accession number. Used by the in-process polling watcher when its diff against `_seen_accessions.json` finds a new submission on EDGAR.

Each downloaded document goes through `_doc_markdown` (datamule's built-in HTML→markdown render) and then `markdown_cleanup.py` (post-processing for two SEC-specific quirks — the `$`+value double-column pattern that confuses fiscal-year reads, and table-cell collapsing). The cleaned files land under `filings/<TICKER>/{10-K,10-Q,8-K}/<period>/` ready for the agent to `grep`.

Datamule handles SEC's rate limits, retry-after headers, and the HTML→markdown conversion of EDGAR's malformed-by-modern-standards filings — the slowest, most thankless parts of any SEC ingest project. The rest of our pipeline (file layout, section splitting, dedup, atomic writes, INDEX.md generation, coverage_index rebuild) is ours.

## Project structure

```
stratalens_ai/
├── fs_research_agent/        # The active agent + tools + corpus management
│   ├── agent.py              # ReAct loop on OpenAI function-calling
│   ├── tools.py              # ls / read_file / grep / glob / news_search
│   ├── prompts.py            # System prompt (date-injected per request)
│   ├── citations.py          # path:line → [10K-N] markers + post-processor
│   ├── orchestrator_adapter.py  # Adapter for the chat router
│   ├── coverage_index.py     # Per-ticker filing index for the UI
│   ├── bootstrap.py          # S3 corpus snapshot upload/download
│   ├── watcher.py            # Background SEC poller
│   ├── observability.py      # Logfire span wrapper
│   ├── tech_universe.json    # Canonical 300+ ticker list (CIK-resolved, regenerated from FinanceDatabase)
│   └── data/                 # Corpus root (gitignored, lives on Railway volume)
├── agent/
│   └── screener/             # DuckDB-backed stock screener
├── app/
│   ├── __init__.py           # FastAPI app + global exception handlers
│   ├── lifespan.py           # Startup/shutdown (auto-bootstrap, watcher spawn)
│   ├── routers/              # API endpoints (chat, coverage, screener, ...)
│   ├── schemas/              # Pydantic request/response models
│   └── utils/                # llm_errors, logfire_config, ...
├── frontend/                 # React + TypeScript + Vite
├── docs/                     # Long-form design docs
├── nixpacks.toml             # Railway build config (installs ripgrep)
├── railway.toml              # Railway deploy config
└── requirements.txt          # Python deps
```

---

## Quick start

### Prerequisites
- Python 3.11+
- `ripgrep` on PATH (`apt-get install ripgrep` or `brew install ripgrep`)
- PostgreSQL 12+ (auth + chat history; not used by the agent itself)
- An OpenAI API key

### Install

```bash
git clone https://github.com/kamathhrishi/finance-agent.git
cd finance-agent
pip install -r requirements.txt
cp .env.example .env   # then fill in keys
```

### Build the corpus (first time only — local dev)

```bash
# Either: download a ready snapshot from S3
python -m fs_research_agent.bootstrap download

# Or: ingest from scratch (slow, hits SEC EDGAR)
python -m fs_research_agent.batch_ingest --years 3
```

### Build frontend + run server

```bash
cd frontend && npm install && npm run build && cd ..
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000` — chat, browse companies, view filings.

---

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | **Yes** | LLM calls (gpt-5.4-mini default) |
| `DATABASE_URL` | **Yes** | Postgres for users + chat history |
| `REDIS_URL` | **Yes** | WebSocket sessions |
| `CLERK_SECRET_KEY` / `CLERK_PUBLISHABLE_KEY` | Production | Auth |
| `TAVILY_API_KEY` | Optional | Enables `news_search` tool. Agent silently skips news without it. |
| `LOGFIRE_TOKEN` | Optional | Observability dashboard |
| `FS_RESEARCH_DATA_ROOT` | Optional | Corpus path. Auto-defaults to `/data/fs_research_corpus` on Railway. |
| `FS_RESEARCH_BOOTSTRAP_FROM_S3` | Optional | Force corpus hydration from S3. Auto-on when Railway + S3 creds detected. |
| `FS_RESEARCH_WATCHER_ENABLED` | Optional | Spawn background SEC watcher. Auto-on when Railway is detected. |
| `DATAMULE_SEC_USER_AGENT` | Required if watcher is on | SEC requires a real `Name email@domain` UA. |
| `FS_RESEARCH_S3_AUTO_UPLOAD_HOURS` | Optional (default 24) | Watcher re-snapshots the corpus to S3 every N hours after meaningful changes. Set to `0` to disable auto-upload. Keeps the bucket within ~24h of the live volume so a future cold-start bootstrap doesn't lose between-snapshot watcher additions. |
| `FS_RESEARCH_S3_AUTO_UPLOAD_MIN_NEW` | Optional (default 50) | Minimum new filings since last upload before re-snapshotting. Prevents wasteful uploads after near-empty cycles. |
| `FS_RESEARCH_BOOTSTRAP_REFRESH_GAP` | Optional (default 500) | Smart-bootstrap threshold. If the local volume already has corpus data but the S3 snapshot is at least N filings ahead, the next deploy re-pulls the fresh snapshot. Set to `0` to disable smart-refresh and revert to "first-time-only" bootstrap semantics. Closes the loop after universe expansions or batch-ingest pushes — no manual volume wipe needed. |
| `RAILWAY_BUCKET_*` | Required for S3 bootstrap | Standard Railway S3 vars (endpoint / key / secret / name) |

---

## API documentation

Auto-generated from the FastAPI app — see the live spec rather than a curated list (which goes stale fast):

- Swagger UI: `/docs` (e.g. `http://localhost:8000/docs`)
- ReDoc: `/redoc`

---

## Documentation

| Document | Description |
|---|---|
| **[fs_research_agent/README.md](fs_research_agent/README.md)** | Deep dive: tools, prompts, citations, ingest, benchmarks, lessons |
| **[Original blog post (chunked RAG approach)](https://substack.com/home/post/p-181608263)** | Historical: the multi-agent / vector-retrieval pipeline that preceded this design |

---

## Deployment

- **Platform**: Railway (Nixpacks builder)
- **Volume**: persistent volume mounted at `/data/fs_research_corpus` for the SEC corpus
- **Cold start**: corpus auto-hydrates from S3 (`s3://<bucket>/fs_research_agent/corpus/latest.tar.gz`) on first boot, then short-circuits on every redeploy
- **Watcher**: in-process asyncio task, polls SEC every 30 min, writes new filings directly to the volume
- **Observability**: Logfire spans across the whole agent lifecycle (flow / llm_round / tool_call / force_final / errors)

See [`fs_research_agent/README.md`](fs_research_agent/README.md) for the operations playbook.

---

## Contributing

Open an issue to discuss major changes before submitting a PR.

## License

MIT — see `LICENSE`.

## Contact

hrishi@stratalens.ai
