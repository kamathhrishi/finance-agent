# Data Ingestion Scripts

Scripts for downloading and processing financial data for the RAG system.
Data is stored in PostgreSQL (pgvector) for embeddings/metadata and Railway S3 for full documents.

## 📁 Script Reference

| Script | Purpose |
|--------|---------|
| `download_transcripts.py` | Download earnings transcripts (API Ninjas, 2020–2025). Saves as JSON. |
| `create_and_store_embeddings.py` | Chunk transcripts, generate embeddings, store in PostgreSQL. |
| `ingest_10k_filings_full.py` | **Core module** (~2,350 lines). Download, parse, chunk, embed SEC filings. Imported by other scripts. |
| `ingest_10k_to_database.py` | Main 10-K ingestion script. Stores to PostgreSQL. Use `--ticker`, `--tickers`, or `--all`. |
| `ingest_sec_filings.py` | Unified ingestion for any filing type (10-K, 10-Q, 8-K). |
| `ingest_with_structure.py` | Structured ingestion that preserves markdown headings and `char_offset` for precise highlighting. Uploads full markdown to S3. |
| `ingest_sp500_10k.py` | Bulk ingestion for all S&P 500 companies. Parallel processing. |

## 🚀 Quick Start (New Setup)

```bash
# 1. Copy env and add keys
cp .env.example .env

# 2. Download transcripts
python agent/rag/data_ingestion/download_transcripts.py

# 3. Create transcript embeddings
python agent/rag/data_ingestion/create_and_store_embeddings.py

# 4. Ingest 10-K filings (single ticker example)
python agent/rag/data_ingestion/ingest_10k_to_database.py --ticker AAPL
```

## ⚙️ Environment Variables

```
DATABASE_URL                  # PostgreSQL + pgvector connection string
API_NINJAS_KEY                # For transcript downloads
RAILWAY_BUCKET_ENDPOINT       # S3-compatible bucket endpoint
RAILWAY_BUCKET_ACCESS_KEY_ID
RAILWAY_BUCKET_SECRET_KEY
RAILWAY_BUCKET_NAME
```

## 💾 Storage Architecture

Data is split across two stores:

- **PostgreSQL**: chunk embeddings, metadata, section structure, `char_offset` positions
- **Railway S3 bucket**: full filing markdown, full transcript text (fetched on demand when user views a document)

New ingestions via `ingest_with_structure.py` automatically upload the full markdown to S3 and store only the `bucket_key` in the DB.

Expected sizes:
- Transcript chunks: ~500MB per 1,000 companies
- 10-K chunks + embeddings: ~5–10GB per 500 companies

## 🔧 Troubleshooting

```bash
# Check if a filing is in the DB and has a bucket key
psql $DATABASE_URL -c "SELECT ticker, filing_type, fiscal_year, bucket_key IS NOT NULL as in_bucket FROM complete_sec_filings WHERE ticker='AAPL' ORDER BY fiscal_year DESC;"
```
