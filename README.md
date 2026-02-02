# StrataLens AI

Stratalens AI is an equity research platform. Ask questions and get answers from 10-K filings, earnings calls, and news.

**Live Platform:** [www.stratalens.ai](https://www.stratalens.ai)

**10K filings agent blogpost:** [Blogpost](https://substack.com/home/post/p-181608263)

## Agent System

Core agent system implementing **Retrieval-Augmented Generation (RAG)** with **semantic data source routing**, **research planning**, and **iterative self-improvement** for financial Q&A.

### Architecture Overview

```
                              AGENT PIPELINE
 ═══════════════════════════════════════════════════════════════════════

 ┌──────────┐    ┌───────────────────┐    ┌──────────────────────────┐
 │ Question │───►│ Question Analyzer │───►│  Semantic Data Routing   │
 └──────────┘    │   (Cerebras LLM)  │    │                          │
                 │                   │    │  • Earnings Transcripts  │
                 │ Extracts:         │    │  • SEC 10-K Filings      │
                 │ • Tickers         │    │  • Real-Time News        │
                 │ • Time periods    │    │  • Hybrid (multi-source) │
                 │ • Intent          │    └────────────┬─────────────┘
                 └───────────────────┘                 │
                                                       ▼
                 ┌─────────────────────────────────────────────────────┐
                 │              RESEARCH PLANNING                       │
                 │  Agent generates reasoning: "I need to find..."     │
                 └────────────────────────┬────────────────────────────┘
                                          ▼
                 ┌─────────────────────────────────────────────────────┐
                 │                  RETRIEVAL LAYER                     │
                 │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
                 │  │  Earnings   │  │  SEC 10-K   │  │   Tavily    │  │
                 │  │ Transcripts │  │   Filings   │  │    News     │  │
                 │  │             │  │             │  │             │  │
                 │  │ Vector DB   │  │ Section     │  │  Live API   │  │
                 │  │ + Hybrid    │  │ Routing +   │  │             │  │
                 │  │   Search    │  │ Reranking   │  │             │  │
                 │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
                 └─────────┴───────────┬────┴────────────────┴─────────┘
                                       │ ▲
                                       │ │ Re-query with
                                       │ │ follow-up questions
                                       ▼ │
                 ┌─────────────────────────────────────────────────────┐
                 │               ITERATIVE IMPROVEMENT                  │
                 │                                                      │
                 │    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
                 │    │ Generate │───►│ Evaluate │───►│ Iterate? │─────┼───┐
                 │    │  Answer  │    │ Quality  │    │          │     │   │
                 │    └──────────┘    └──────────┘    └──────────┘     │   │
                 │                                         │ NO        │   │ YES
                 └─────────────────────────────────────────┼───────────┘   │
                                                           ▼               │
                                                    ┌─────────────┐        │
                                                    │   ANSWER    │        │
                                                    │ + Citations │        │
                                                    └─────────────┘        │
                                                           ▲               │
                                                           └───────────────┘
```

**Key Concepts:**
1. **Semantic Routing** - Routes to data sources based on question **intent**, not just keywords
2. **Research Planning** - Agent explains reasoning before searching ("I need to find...")
3. **Multi-Source RAG** - Combines earnings transcripts, SEC 10-K filings, and news
4. **Self-Reflection** - Evaluates answer quality and iterates until confident (≥90%)

**Benchmark:** 91% accuracy on [FinanceBench](https://github.com/patronus-ai/financebench) (112 10-K questions), ~10s per question, evaluated using LLM-as-a-judge.

### Documentation

| Document | Description |
|----------|-------------|
| **[agent/README.md](agent/README.md)** | Complete agent architecture, pipeline stages, configuration |
| **[docs/SEC_AGENT.md](docs/SEC_AGENT.md)** | SEC 10-K agent: section routing, table selection, reranking |
| **[agent/rag/data_ingestion/README.md](agent/rag/data_ingestion/README.md)** | Data ingestion pipelines for transcripts and 10-K filings |

---

## Features

- **Earnings Transcripts** (2022-2025) - Word-for-word executive commentary
- **SEC Filings** (10K of 2024-25) - Official 10-K and 10-Q reports
- **Financial Screener** - Natural language queries over company fundamentals [not in production]

Unlike generic LLMs that rely on web content, StrataLens uses the same authoritative documents that professional analysts depend on.

## Tech Stack

- **Backend:** FastAPI, PostgreSQL (pgvector), DuckDB
- **AI/ML:** Cerebras (Qwen-3-235B), OpenAI (fallback), RAG with iterative self-improvement
- **Search:** Hybrid vector (pgvector) + TF-IDF with cross-encoder reranking
- **Frontend:** React + TypeScript, Tailwind CSS

## Project Structure

```
stratalens_ai/
├── agent/                  # AI agent & RAG system         → see agent/README.md
│   ├── rag/               # RAG implementation
│   │   ├── rag_agent.py            # Main orchestration (2,700+ lines)
│   │   ├── sec_filings_service_smart_parallel.py  # 10-K agent (current)
│   │   ├── sec_filings_service_iterative.py       # 10-K agent (legacy)
│   │   ├── response_generator.py   # LLM response & evaluation
│   │   ├── question_analyzer.py    # Semantic routing
│   │   ├── search_engine.py        # Hybrid transcript search
│   │   ├── tavily_service.py       # Real-time news
│   │   └── data_ingestion/         # Data pipeline → see data_ingestion/README.md
│   └── screener/          # Financial screener
├── app/                   # FastAPI application
│   ├── routers/           # API endpoints
│   ├── schemas/           # Pydantic models
│   └── auth/              # Authentication
├── frontend/              # React + TypeScript frontend
├── docs/                  # Documentation
│   └── SEC_AGENT.md       # 10-K agent deep dive
├── analytics/             # Usage analytics
└── experiments/           # Development & benchmarking (gitignored)
    ├── sec_filings_rag_scratch/   # Agent evolution & benchmark results
    ├── sec_filings_rag/           # Hierarchical parsing experiments
    └── llamaindex_agent/          # LlamaIndex alternative approach
```

## Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL 12+ with pgvector extension
- See [Requirements](#requirements) for full dependency list

### Installation

```bash
# Clone repository
git clone https://github.com/kamathhrishi/stratalensai.git
cd stratalens_ai

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys and database credentials

# Configure environment (see Configuration section below)
```

### Configuration

Before running the application, configure the following files based on your environment:

**Backend (`.env`):**
- `BASE_URL` - Set to your server URL (e.g., `localhost:8000` for local, your production URL for deployment)
- `RAG_DEBUG_MODE` - Set to `false` for production, `true` for development debugging
- `ENABLE_LOGIN` / `ENABLE_SELF_SERVE_REGISTRATION` - Toggle authentication features as needed

**Frontend (`frontend/config.js`):**
- `ENVIRONMENT` - Set to `'local'` for development or `'production'` for deployment
- Update the `ENVIRONMENTS.production` URLs to match your production server

```bash
# Initialize database
python utils/database_init.py

# Ingest data (optional - see data_ingestion/README.md)
python agent/rag/data_ingestion/create_tables.py
python agent/rag/data_ingestion/download_transcripts.py

# Run server
python fastapi_server.py
```

Access the application at `http://localhost:8000`

## Requirements

### API Keys

| Service | Environment Variable | Required |
|---------|---------------------|----------|
| OpenAI | `OPENAI_API_KEY` | Yes |
| Cerebras | `CEREBRAS_API_KEY` | Yes |
| API Ninjas | `API_NINJAS_KEY` | Yes |
| Tavily | `TAVILY_API_KEY` | Optional |
| Logfire | `LOGFIRE_TOKEN` | Optional |
| Google OAuth | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Optional |

### Database

- **PostgreSQL** with [pgvector](https://github.com/pgvector/pgvector) extension (`DATABASE_URL`)
- **Redis** (optional, for caching) (`REDIS_URL`)

### Python Dependencies

See `requirements.txt` for full list.

## API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key Endpoints

- `POST /message/stream-v2` - Chat with streaming RAG responses
- `GET /companies/search` - Search companies by ticker/name
- `GET /transcript/{ticker}/{year}/{quarter}` - Get specific earnings transcript
- `POST /screener/query/stream` - Natural language financial queries

## Data Sources

All downloaded data is stored in `agent/rag/data_downloads/` (gitignored):
- Earnings transcripts (~1-2GB per 1000 companies)
- Vector embeddings (~500MB per 1000 companies)
- SEC filings (~5-10GB per 500 companies)

See `agent/rag/data_ingestion/README.md` for detailed ingestion instructions.

## AI Agent Documentation

For detailed documentation on the AI agent architecture and RAG system:

| Document | Description |
|----------|-------------|
| **[agent/README.md](agent/README.md)** | Complete agent architecture, 6-stage pipeline, semantic routing, iterative self-improvement |
| **[docs/SEC_AGENT.md](docs/SEC_AGENT.md)** | SEC 10-K agent: SmartParallel architecture, planning-driven retrieval, 91% accuracy |
| **[agent/rag/data_ingestion/README.md](agent/rag/data_ingestion/README.md)** | Data ingestion pipelines for transcripts, embeddings, and SEC filings |

### Experiments & Development History

The agent evolved through extensive experimentation (in `experiments/` folder):

| Experiment | Description |
|------------|-------------|
| `sec_filings_rag_scratch/` | Original 10-K agent development, benchmark results, optimization analysis |
| `sec_filings_rag/` | Hierarchical parsing experiments |
| `llamaindex_agent/` | Alternative LlamaIndex-based implementation |
| `benchmarks/` | FinanceBench evaluation framework |

## Development Status

**Production (stratalens.ai):**
- Earnings transcript chat with RAG
- SEC 10-K filings (2024-25)
- Real-time streaming responses
- User authentication

**In Development:**
- Enhanced financial screener
- Performance optimizations

## Contributing

Contributions welcome! Please open an issue to discuss major changes before submitting PRs.

## License

MIT License - see LICENSE file for details

## Contact

For questions or access requests: hrishi@stratalens.ai





