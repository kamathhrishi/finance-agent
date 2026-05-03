# StrataLens AI

Equity-research platform that answers questions about public-company SEC filings (10-K · 10-Q · 8-K). Built around a single research agent that reads filings directly off a local filesystem corpus — no chunking, no vector DB, no semantic routing.

**Live platform:** [www.stratalens.ai](https://www.stratalens.ai)

**Coverage:** 138 tech companies · 12,000+ filings · 3 years history. Updated automatically by an in-process watcher polling SEC EDGAR.

---

## Architecture

```
                            CHAT REQUEST
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ FilesystemResearchAgent │
                    │   (gpt-5.4-mini)        │
                    └────────────┬────────────┘
                                 │ ReAct loop
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
       ┌─────────┐         ┌──────────┐         ┌──────────┐
       │   ls    │         │ read_file│         │   grep   │
       │  glob   │         │          │         │ (ripgrep)│
       └────┬────┘         └────┬─────┘         └────┬─────┘
            │                   │                    │
            └────────────┬──────┴────────────────────┘
                         ▼
        ┌────────────────────────────────────────┐
        │  Local SEC corpus (markdown filesystem)│
        │  filings/<TICKER>/{10-K,10-Q,8-K}/...  │
        │  Hosted on Railway persistent volume.  │
        │  Hydrated from S3 on cold start.       │
        └────────────────────────────────────────┘

        + 1 supplemental tool: news_search (Tavily) for post-filing color
        + Background watcher: polls SEC every 30 min, writes to the volume
```

### Why this shape

The agent has four primary tools (`ls`, `read_file`, `grep`, `glob`) over a sandboxed filesystem of pre-cleaned SEC markdown. No embeddings, no vector store, no rerank pipeline. The LLM does its own retrieval the same way an analyst would: list the directory, read the index, grep for the figure, quote the line.

This replaces an earlier multi-agent / chunk-RAG pipeline (semantic-routing → vector retrieval → cross-encoder rerank → iterative self-improvement) — see the original blog post on that approach: [Building a 10-K research agent with chunked RAG](https://substack.com/home/post/p-181608263). It worked, but the new model generation made the simpler harness measurably better at long-form filings reasoning. Design notes in [`fs_research_agent/README.md`](fs_research_agent/README.md).

---

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
│   ├── tech_universe.json    # Canonical 138-ticker list (CIK-resolved)
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
git clone https://github.com/kamathhrishi/stratalensai.git
cd stratalensai
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
| `RAILWAY_BUCKET_*` | Required for S3 bootstrap | Standard Railway S3 vars (endpoint / key / secret / name) |

---

## API documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat/message/stream-v2` | Chat (authenticated, streaming SSE) |
| `POST` | `/chat/landing/demo/stream-v2` | Chat (anonymous demo, streaming SSE) |
| `GET` | `/coverage/status` | Universe stats + last-refresh time |
| `GET` | `/coverage/companies` | All companies + per-form filing counts |
| `GET` | `/coverage/companies/{ticker}` | Drill-down: every filing for one ticker |
| `GET` | `/coverage/latest` | Newest-first feed of filings (paginated) |
| `GET` | `/fs-research/document` | Raw filing markdown by path |
| `POST` | `/fs-research/document/with-highlights` | Filing markdown with line-range highlights for citation rendering |

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
