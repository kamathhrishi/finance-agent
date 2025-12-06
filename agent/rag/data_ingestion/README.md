# Data Ingestion Scripts

Scripts for downloading and processing financial data for the RAG system.

## 📁 Structure

```
agent/rag/
├── data_downloads/              # ← Downloaded data (GITIGNORED)
│   ├── transcripts/            # Earnings transcripts
│   ├── embeddings/             # Embeddings cache
│   ├── 10k_filings/            # 10-K SEC filings
│   ├── duckdb/                 # DuckDB files
│   └── cache/                  # Temp cache
│
└── data_ingestion/             # ← Scripts (committed)
    ├── download_transcripts.py
    ├── ingest_10k_filings.py
    └── ...more scripts
```

## 🚀 Quick Start

```bash
# 1. Setup
cp .env.example .env
# Add your API keys to .env

# 2. Create database tables
python agent/rag/data_ingestion/create_tables.py

# 3. Download data (saves to data_downloads/)
python agent/rag/data_ingestion/download_transcripts.py
python agent/rag/data_ingestion/ingest_10k_filings.py
```

## 📋 Scripts

**Core:**
- `download_transcripts.py` - Download earnings transcripts
- `create_and_store_embeddings.py` - Generate embeddings
- `ingest_10k_filings.py` - Download 10-K filings
- `ingest_10k_filings_full.py` - Full 10-K ingestion
- `ingest_sp500_10k.py` - S&P 500 10-Ks

**Database:**
- `create_tables.py` - Create PostgreSQL tables
- `drop_tables.py` - Drop tables
- `test_db_connection.py` - Test connection

**Utilities:**
- `fetch_us_tickers.py` - Get ticker list
- `fetch_finqual.py` - Financial metrics

## ⚙️ Environment Variables

Required in `.env`:
- `OPENAI_API_KEY` - For embeddings
- `API_NINJAS_KEY` - For transcripts
- `DATABASE_URL` - PostgreSQL connection

## 💾 Storage

All data saves to `agent/rag/data_downloads/` (gitignored)

Expected sizes:
- Transcripts: ~1-2GB per 1000 companies
- Embeddings: ~500MB per 1000 companies
- 10-K Filings: ~5-10GB per 500 companies

**Total: 10-20GB for full dataset**

## 🔧 Troubleshooting

```bash
# Test database
python test_db_connection.py

# Check disk space
du -sh ../data_downloads/

# View logs
tail -f *.log
```
