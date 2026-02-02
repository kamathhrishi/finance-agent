# Agent System

Core agent system implementing **Retrieval-Augmented Generation (RAG)** with **semantic data source routing**, **research planning**, and **iterative self-improvement** for financial Q&A. This powers the chat and analysis features on stratalens.ai.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Complete Pipeline](#complete-pipeline)
- [Semantic Data Source Routing](#semantic-data-source-routing)
- [Question Planning & Reasoning](#question-planning--reasoning)
- [Self-Reflection Loop](#self-reflection-loop)
- [Data Sources](#data-sources)
  - [Earnings Transcripts](#earnings-transcript-search)
  - [SEC 10-K Filings](#sec-10-k-filings-agent)
  - [Real-Time News](#tavily-real-time-news)
- [Multi-Ticker Synthesis](#multi-ticker-synthesis)
- [Streaming Events](#streaming-events)
- [Configuration](#configuration)
- [Usage](#usage)
- [Key Components](#key-components)

---

## Architecture Overview

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
1. **Semantic Routing** - Routes to data sources based on question **intent**, not keywords
2. **Research Planning** - Agent explains reasoning before searching ("I need to find...")
3. **Multi-Source RAG** - Combines earnings transcripts, SEC filings, and news
4. **Self-Reflection** - Evaluates answer quality and iterates until confident (≥90%)

---

## Complete Pipeline

The agent executes a **6-stage pipeline** for each question:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: SETUP & INITIALIZATION                                          │
│ • Initialize RAG components (search engine, response generator)          │
│ • Load configuration and available quarters                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: QUESTION ANALYSIS (Cerebras LLM)                                │
│ • Extract company tickers ($AAPL, $MSFT)                                 │
│ • Detect time periods (Q4 2024, last 3 quarters, latest)                 │
│ • Semantic routing → Choose data source based on INTENT                  │
│ • Generate semantically-grounded search query                            │
│ • Validate question (reject off-topic/invalid)                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2.1: QUESTION PLANNING/REASONING (NEW)                             │
│ • Agent generates research approach reasoning                            │
│ • Example: "The user is asking about Azure revenue, so I need to find    │
│   quarterly growth rates, management commentary on cloud competition..." │
│ • Streamed to frontend as 'reasoning' event                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2.5: NEWS SEARCH (if needs_latest_news=true)                       │
│ • Query Tavily API for real-time news                                    │
│ • Format with [N1], [N2] citation markers                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2.6: 10-K SEC SEARCH (if data_source="10k" or needs_10k=true)      │
│ • LLM-based section routing (Item 1, Item 7, Item 8, etc.)               │
│ • Hybrid search (TF-IDF + semantic)                                      │
│ • Cross-encoder reranking                                                │
│ • LLM-based table selection                                              │
│ • Format with [10K1], [10K2] citation markers                            │
│ • Uses more iterations (5 vs 4) for thorough analysis                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: TRANSCRIPT SEARCH (Vector + Keyword Hybrid)                     │
│ • Single-ticker: Direct search with quarter filtering                    │
│ • Multi-ticker: Parallel search per company                              │
│ • Hybrid scoring: 70% vector + 30% keyword                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: INITIAL ANSWER GENERATION                                       │
│ • Single ticker → generate_openai_response()                             │
│ • Multiple tickers → generate_multi_ticker_response() with synthesis     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: ITERATIVE IMPROVEMENT (3-5 iterations)                          │
│ For each iteration:                                                      │
│   1. Evaluate answer quality (confidence, completeness, specificity)     │
│   2. Check if reasoning goals are met                                    │
│   3. Generate follow-up questions for gaps                               │
│   4. Search in parallel with follow-up questions                         │
│   5. Agent may request news/transcript search                            │
│   6. Regenerate answer with expanded context                             │
│ Stop when: confidence ≥90%, max iterations, or agent decides sufficient  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: FINAL RESPONSE ASSEMBLY                                         │
│ • Stream final answer with citations                                     │
│ • Include all source attributions (transcripts, 10-K, news)              │
│ • Return metadata (confidence, chunks used, timing)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Semantic Data Source Routing

The agent routes questions based on **intent**, not just keywords. This is a key differentiator from simple keyword matching.

### How It Works

The Question Analyzer uses Cerebras LLM to understand what type of information would **best answer** the question:

```
QUESTION INTENT → DATA SOURCE DECISION

┌─────────────────────────────────────────────────────────────────────────┐
│ 10-K SEC FILINGS (data_source="10k")                                     │
│ Best for:                                                                │
│ • Annual/full-year financial data, audited figures                       │
│ • Balance sheets, income statements, cash flow statements                │
│ • Executive compensation, CEO pay, stock awards (ONLY in 10-K!)          │
│ • Risk factors, legal proceedings, regulatory matters                    │
│ • Detailed business descriptions, segment breakdowns                     │
│ • Multi-year historical comparisons                                      │
│ • Total assets, liabilities, debt structure                              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ EARNINGS TRANSCRIPTS (data_source="earnings_transcripts")                │
│ Best for:                                                                │
│ • Quarterly performance discussions, recent quarter results              │
│ • Management commentary, executive statements, tone/sentiment            │
│ • Forward guidance, outlook, projections                                 │
│ • Analyst Q&A, investor concerns, management responses                   │
│ • Product launches, strategic initiatives                                │
│ • Quarter-over-quarter comparisons                                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ LATEST NEWS (data_source="latest_news")                                  │
│ Best for:                                                                │
│ • Very recent events (last few days/weeks)                               │
│ • Breaking developments, announcements                                   │
│ • Market reactions, stock movements                                      │
│ • Recent partnerships, acquisitions, leadership changes                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ HYBRID (data_source="hybrid")                                            │
│ Best for:                                                                │
│ • Questions explicitly requesting multiple perspectives                  │
│ • Comparing official filings with recent developments                    │
│ • Comprehensive analysis needing historical + current data               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Routing Decision Process

The LLM considers:
1. **Intent**: What is the user trying to learn?
2. **Time Period**: Annual=10K, Quarterly=Transcripts, Recent=News
3. **Formality**: Official/Audited=10K, Commentary=Transcripts, Current=News
4. **Completeness**: Would combining sources provide a better answer?

### Examples

| Question | Routed To | Reasoning |
|----------|-----------|-----------|
| "What was Apple's Q4 2024 revenue?" | Transcripts | Quarterly data, recent results |
| "What is Tim Cook's compensation?" | 10-K | Executive compensation only in SEC filings |
| "Show me Microsoft's balance sheet" | 10-K | Financial statements from annual reports |
| "What did management say about AI?" | Transcripts | Management commentary from earnings calls |
| "What's the latest news on NVIDIA?" | News | Recent developments |
| "Compare 10-K risks with recent news" | Hybrid | Needs multiple sources |

---

## Question Planning & Reasoning

**New Feature**: Before searching, the agent generates a reasoning statement explaining its research approach.

### Purpose

- Makes the agent's thinking transparent
- Guides evaluation (did we find what we planned to find?)
- Improves answer quality through structured research

### Example

```
User: "What is Microsoft's cloud strategy and how is Azure performing?"

Agent Reasoning:
"The user is asking about Microsoft's cloud business strategy and Azure
performance. I need to find:
- Azure revenue figures and growth rates (quarterly)
- Management commentary on competitive positioning vs AWS/Google Cloud
- Margin trends and profitability metrics
- Forward guidance for cloud segment
Key metrics: quarterly revenue, YoY growth %, operating margins.
I'll focus on the most recent quarters available and look for strategic
commentary from executives."
```

### Implementation

```python
# From prompts.py
QUESTION_PLANNING_SYSTEM_PROMPT = """You are a financial research analyst
who thinks through questions before searching. You explain your reasoning
process in a natural, verbose way - like thinking out loud about how to
approach a research question."""

# Generates 3-5 sentence reasoning explaining:
# - What the user is really trying to understand
# - What specific metrics/data points needed
# - What to focus the search on
# - How to approach given available data
```

---

## Self-Reflection Loop

The agent performs iterative self-improvement until the answer meets quality thresholds.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ITERATION LOOP                                │
│                                                                  │
│  ┌──────────────────┐                                           │
│  │ Generate Answer  │◄──────────────────────────────────┐       │
│  └────────┬─────────┘                                   │       │
│           │                                             │       │
│           ▼                                             │       │
│  ┌──────────────────┐                                   │       │
│  │ Evaluate Quality │                                   │       │
│  │ • completeness   │                                   │       │
│  │ • specificity    │                                   │       │
│  │ • accuracy       │                                   │       │
│  │ • vs. reasoning  │ ← Checks if reasoning goals met   │       │
│  └────────┬─────────┘                                   │       │
│           │                                             │       │
│           ▼                                             │       │
│  ┌──────────────────┐    YES    ┌─────────────────┐    │       │
│  │ Confidence < 90% │─────────► │ Search for more │────┘       │
│  │ & iterations left│           │ context (tools) │            │
│  └────────┬─────────┘           └─────────────────┘            │
│           │ NO                                                  │
│           ▼                                                     │
│     ┌───────────┐                                               │
│     │  OUTPUT   │                                               │
│     └───────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

**Evaluation Scores (0-100):**
- `completeness_score`: Does the answer fully address the question?
- `specificity_score`: Does it include specific numbers, quotes?
- `accuracy_score`: Is the information factually correct?
- `clarity_score`: Is the response well-structured?
- `overall_confidence` (0-1): Weighted combination

**During iteration, the agent can:**
- Generate follow-up questions (searched in parallel)
- Request additional transcript search (`needs_transcript_search`)
- Request news search (`needs_news_search`)

**Stops when:**
1. Confidence ≥ 90%
2. Max iterations reached (3 for general, 5 for 10-K)
3. Agent decides answer is sufficient
4. No follow-up questions generated

---

## Data Sources

### Earnings Transcript Search

For quarterly earnings questions, uses hybrid search over transcript chunks.

```python
# search_engine.py
def search_similar_chunks(query, top_k, quarter):
    """
    Hybrid search combining:
    - Vector search: 70% weight (semantic similarity via pgvector)
    - Keyword search: 30% weight (TF-IDF)
    """
```

**Database Schema:**
```
PostgreSQL Table: transcript_chunks
├── chunk_text: TEXT (1000 chars max, 200 overlap)
├── embedding: VECTOR (all-MiniLM-L6-v2, 384 dimensions)
├── ticker: VARCHAR (e.g., "AAPL")
├── year: INTEGER (e.g., 2024)
├── quarter: INTEGER (1-4)
└── metadata: JSONB
```

---

### SEC 10-K Filings Agent

**Dedicated documentation:** [docs/SEC_AGENT.md](../docs/SEC_AGENT.md)

The SEC agent retrieves data from 10-K filings using planning-driven parallel retrieval.

**Benchmark:** 91% accuracy on FinanceBench (112 questions), ~10s per question

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         10-K SEARCH FLOW (max 5 iterations)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │ PHASE 0: PLAN   │   Generate sub-questions + search plan                │
│  │ • Sub-questions │   "What is inventory turnover?" →                     │
│  │ • Search plan   │     - "What is COGS?" [TABLE]                         │
│  └────────┬────────┘     - "What is inventory?" [TABLE]                    │
│           │              - "Inventory valuation?" [TEXT]                   │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 1: PARALLEL RETRIEVAL                                         │   │
│  │ ├── Execute ALL searches in parallel (6 workers)                    │   │
│  │ │   ├── TABLE: "cost of goods sold" → LLM selects tables            │   │
│  │ │   ├── TABLE: "inventory balance" → LLM selects tables             │   │
│  │ │   └── TEXT: "inventory valuation" → hybrid search                 │   │
│  │ └── Deduplicate and combine chunks                                  │   │
│  └────────┬────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │ PHASE 2: ANSWER │   Generate answer with ALL retrieved chunks          │
│  └────────┬────────┘                                                        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │ PHASE 3: EVAL   │   If quality >= 90% → DONE                            │
│  │                 │   Else → Replan and loop back                         │
│  └─────────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Generates targeted sub-questions for retrieval (not just original question)
- Parallel search execution for speed
- Dynamic replanning based on evaluation gaps

**Switching Versions:**
```python
# Current:
from .sec_filings_service_smart_parallel import SmartParallelSECFilingsService as SECFilingsService

# Iterative (legacy):
from .sec_filings_service_iterative import IterativeSECFilingsService as SECFilingsService
```

---

### Tavily (Real-Time News)

`tavily_service.py` provides real-time web search for current events.

**When Used:**
1. Question contains news keywords ("latest news", "recent developments")
2. Agent requests during iteration (`needs_news_search=true`)

**How It Works:**
```python
class TavilyService:
    def search_news(self, query: str, max_results: int = 5):
        """
        Returns:
            {
                "answer": "AI-generated summary",
                "results": [
                    {
                        "title": "Article headline",
                        "url": "https://...",
                        "content": "Article text",
                        "published_date": "2024-01-15"
                    }
                ]
            }
        """

    def format_news_context(self, news_results):
        """Formats with [N1], [N2] citation markers"""
```

---

## Multi-Ticker Synthesis

For questions comparing multiple companies, the agent:

1. **Parallel Processing**: Searches each ticker concurrently
2. **Ticker-Specific Rephrasing**: Creates company-specific search queries
3. **Synthesis**: Combines results into unified comparative analysis

```
Input: "Compare $AAPL and $MSFT revenue"

Process:
├── Rephrase for AAPL: "revenue and sales performance"
├── Rephrase for MSFT: "revenue and sales performance"
├── Search AAPL chunks (parallel)
├── Search MSFT chunks (parallel)
├── Synthesis prompt combines both
└── Output: Comparative analysis with both companies

Synthesis Requirements:
• ALWAYS maintain period metadata (Q1 2025, FY 2024)
• ALWAYS include ALL financial figures from ALL sources
• Show trends and comparisons across companies
• Use human-friendly format: "Q1 2025" not "2025_q1"
```

---

## Streaming Events

The agent streams real-time progress events to the frontend:

| Event Type | Description |
|------------|-------------|
| `progress` | Generic progress updates |
| `analysis` | Question analysis complete |
| `reasoning` | Agent's research planning statement |
| `news_search` | News search results |
| `10k_search` | 10-K SEC search results |
| `iteration_start` | Beginning of iteration N |
| `agent_decision` | Agent's quality assessment |
| `iteration_followup` | Follow-up questions being searched |
| `iteration_search` | New chunks found |
| `iteration_complete` | Iteration finished |
| `result` | Final answer with citations |
| `rejected` | Question rejected (out of scope) |
| `error` | Error occurred |

**Event Structure:**
```json
{
  "type": "reasoning",
  "message": "The user is asking about Microsoft's cloud strategy...",
  "step": "planning",
  "data": {
    "reasoning": "Full reasoning statement..."
  }
}
```

---

## Configuration

### Environment Variables

```bash
OPENAI_API_KEY=...           # Response generation (fallback)
CEREBRAS_API_KEY=...         # Question analysis, routing, planning
TAVILY_API_KEY=...           # Real-time news search
DATABASE_URL=postgresql://...# Main database
PG_VECTOR=postgresql://...   # Vector search database
LOGFIRE_TOKEN=...            # Observability (optional)
```

### Agent Config (`agent_config.py`)

```python
{
    "max_iterations": 4,              # General questions
    "sec_max_iterations": 5,          # 10-K questions (more thorough)
    "min_confidence_threshold": 0.90, # High bar for early stopping
    "min_completeness_threshold": 0.90,
}
```

### RAG Config (`rag/config.py`)

```python
{
    "chunks_per_quarter": 15,         # Results per quarter
    "max_quarters": 12,               # Max 3 years of data
    "max_tickers": 8,                 # Max companies per query

    # Hybrid search weights
    "keyword_weight": 0.3,
    "vector_weight": 0.7,

    # Models
    "cerebras_model": "qwen-3-235b-a22b-instruct-2507",
    "openai_model": "gpt-4.1-mini-2025-04-14",
    "embedding_model": "all-MiniLM-L6-v2",
}
```

---

## Usage

```python
from agent import create_agent

agent = create_agent()

# Earnings transcript question (automatic routing)
async for event in agent.execute_rag_flow(
    question="What did $AAPL say about iPhone sales in Q4 2024?",
    stream=True
):
    if event['type'] == 'reasoning':
        print(f"Planning: {event['message']}")
    elif event['type'] == 'result':
        print(f"Answer: {event['data']['answer']}")

# 10-K question (automatically routes to SEC filings)
result = await agent.execute_rag_flow_async(
    question="What was Tim Cook's compensation in 2023?"
)

# News question (automatically routes to Tavily)
result = await agent.execute_rag_flow_async(
    question="What's the latest news on $NVDA?"
)

# Multi-ticker comparison
async for event in agent.execute_rag_flow(
    question="Compare $MSFT and $GOOGL cloud revenue",
    stream=True,
    max_iterations=4
):
    print(event)
```

---

## Key Components

### Core Files

| File | Description |
|------|-------------|
| `agent.py` | Main entry point - unified Agent API |
| `agent_config.py` | Agent configuration and iteration settings |
| `prompts.py` | Centralized LLM prompt templates (including planning) |
| `rag/rag_agent.py` | Orchestration engine with pipeline stages |
| `rag/question_analyzer.py` | LLM-based semantic routing (Cerebras) |

### Data Sources (Tools)

| File | Tool | Description |
|------|------|-------------|
| `rag/search_engine.py` | Transcript Search | Hybrid vector + keyword search |
| `rag/sec_filings_service_smart_parallel.py` | 10-K Search | Planning + parallel retrieval (default) |
| `rag/sec_filings_service_iterative.py` | 10-K Search | Iterative table/text decisions (legacy) |
| `rag/tavily_service.py` | News Search | Real-time news via Tavily API |

### Supporting Components

| File | Description |
|------|-------------|
| `rag/response_generator.py` | LLM response generation, evaluation, planning |
| `rag/database_manager.py` | PostgreSQL/pgvector operations |
| `rag/conversation_memory.py` | Multi-turn conversation state |
| `rag/config.py` | RAG configuration |

---

## Database Schema

```
PostgreSQL + pgvector
├── transcript_chunks       # Earnings call transcripts
│   ├── chunk_text          # 1000 chars, 200 overlap
│   ├── embedding           # all-MiniLM-L6-v2 (384 dim)
│   ├── ticker, year, quarter
│   └── metadata (JSONB)
│
├── ten_k_chunks            # 10-K filing text
│   ├── chunk_text, embedding
│   ├── sec_section         # item_1, item_7, item_8, etc.
│   ├── sec_section_title   # Human-readable section name
│   └── is_financial_statement
│
└── ten_k_tables            # 10-K extracted tables (JSONB)
    ├── content             # Table data
    ├── statement_type      # income_statement, balance_sheet, cash_flow
    └── is_financial_statement
```

---

## Limitations

- Requires `$TICKER` format for company identification
- Quarter availability varies by company
- Companies describe fiscal years differently
- No real-time stock price data
- 10-K data limited to 2024-25 filings currently

---

## Development Status

| Component | Status |
|-----------|--------|
| Semantic Data Source Routing | ✅ Production |
| Question Planning/Reasoning | ✅ Production |
| Earnings Transcript Search | ✅ Production |
| SEC 10-K Filing Search | ✅ Production (91% accuracy on FinanceBench) |
| Tavily News Search | ✅ Production |
| Multi-ticker Synthesis | ✅ Production |
| Iterative Improvement | ✅ Production |
| Streaming Events | ✅ Production |
| Conversation Memory | ✅ Production |

---

## Related Documentation

- **[Main README](../README.md)** - Project overview and setup
- **[SEC Agent](../docs/SEC_AGENT.md)** - Detailed 10-K agent: section routing, LLM table selection, cross-encoder reranking
- **[SEC RAG Experiments](../experiments/sec_filings_rag/README.md)** - Hierarchical parsing prototype
- **[Data Ingestion](rag/data_ingestion/README.md)** - Transcript and 10-K ingestion pipelines
