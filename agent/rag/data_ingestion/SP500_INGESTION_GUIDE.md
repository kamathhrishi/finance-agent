# S&P 500 10-K Ingestion Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install datamule scikit-learn pandas openpyxl requests sentence-transformers psycopg2-binary
```

**IMPORTANT: Python 3.11+ Recommended**
- Python 3.11+ includes better memory management features
- Each ticker gets a fresh process (prevents memory accumulation)
- For Python < 3.11, workers are reused but include aggressive garbage collection

### 2. Set Environment Variables
Ensure your `.env` file has the database connection:
```bash
# In your .env file
DATABASE_URL=postgresql://user:password@host:port/database
# OR
PG_VECTOR=postgresql://user:password@host:port/database
```

### 3. Run the Script

**RECOMMENDED: Parallel processing with 5 workers**
```bash
python agent/rag/data_ingestion/ingest_sp500_10k.py --workers 5
```

This will:
1. Automatically fetch the current S&P 500 ticker list (~500 companies)
2. Download and process 10-K filings from the last 1 year
3. Store data in PostgreSQL database with sophisticated processing
4. Complete in ~2-4 hours

## Usage Examples

### Full S&P 500 Ingestion (Recommended)
```bash
# With 5 workers (fastest, ~2-4 hours)
python agent/rag/data_ingestion/ingest_sp500_10k.py --workers 5

# With 3 workers (slower, ~3-6 hours)
python agent/rag/data_ingestion/ingest_sp500_10k.py --workers 3
```

### Testing (Process subset)
```bash
# Test with first 10 companies
python agent/rag/data_ingestion/ingest_sp500_10k.py --max-tickers 10 --workers 3

# Test with first 50 companies
python agent/rag/data_ingestion/ingest_sp500_10k.py --max-tickers 50 --workers 5
```

### Resume from specific position
```bash
# Resume from ticker #200
python agent/rag/data_ingestion/ingest_sp500_10k.py --skip-first 200 --workers 5

# Process tickers 100-200
python agent/rag/data_ingestion/ingest_sp500_10k.py --skip-first 100 --max-tickers 100 --workers 5
```

### Custom time period
```bash
# Get 2 years of data instead of 1
python agent/rag/data_ingestion/ingest_sp500_10k.py --lookback-years 2 --workers 5

# Get 3 years of data
python agent/rag/data_ingestion/ingest_sp500_10k.py --lookback-years 3 --workers 5
```

### Sequential mode (no parallel processing)
```bash
# Slower but simpler (not recommended for full S&P 500)
python agent/rag/data_ingestion/ingest_sp500_10k.py --sequential
```

### Save ticker list
```bash
# Save the fetched S&P 500 ticker list to sp500_tickers.txt
python agent/rag/data_ingestion/ingest_sp500_10k.py --save-tickers --workers 5
```

## What Gets Ingested

For each S&P 500 company, the script:

1. **Fetches 10-K filings** from the last 1 year (or specified period)
2. **Processes with sophisticated extraction**:
   - Hierarchical content extraction (preserves document structure)
   - Section identification (Item 1, Item 7, MD&A, etc.)
   - Table extraction and financial statement identification
   - Contextual chunking for better RAG retrieval

3. **Stores in PostgreSQL**:
   - `ten_k_chunks` table: Text chunks with embeddings and metadata
   - `ten_k_tables` table: Extracted tables with financial statement classification

## Database Tables

### `ten_k_chunks`
Stores chunked text segments with vector embeddings:
- `chunk_text`: Text content
- `embedding`: Vector(384) embedding
- `ticker`: Stock ticker
- `fiscal_year`: Fiscal year
- `sec_section`: Section ID (item_1, item_7, etc.)
- `sec_section_title`: Human-readable section name
- `path_string`: Hierarchical path in document
- `metadata`: Full JSONB metadata

### `ten_k_tables`
Stores extracted tables:
- `content`: Table content as text
- `ticker`: Stock ticker
- `fiscal_year`: Fiscal year
- `is_financial_statement`: Boolean flag
- `statement_type`: income_statement, balance_sheet, cash_flow
- `priority`: CRITICAL for key financial statements

## Performance

| Configuration | Time Estimate | Recommended For |
|--------------|---------------|-----------------|
| 5 workers | 2-4 hours | Full S&P 500 ingestion |
| 3 workers | 3-6 hours | Standard processing |
| Sequential | 10-15 hours | Testing/debugging only |

**Per company**: ~1-3 minutes (with data), ~10-30 seconds (no data)

## Progress Tracking

The script provides detailed progress updates:

```
⏱️  PROGRESS UPDATE
Completed: 100/503 (19.9%)
Successful: 87
Time elapsed: 45.2m
Estimated remaining: 181.3m
```

Progress updates occur every 10 completions.

## Success Metrics

At the end, you'll see:

```
📊 FINAL SUMMARY
Total execution time: 156.3 minutes (2.61 hours)
Tickers processed: 503

📈 DATA INGESTED:
  Total 10-K filings: 489
  Total text chunks: 1,234,567
  Total tables extracted: 45,678
  Total Finqual statements: 1,467

📊 RESULTS BREAKDOWN:
  ✅ Successful: 487 companies
  ⚠️  No data found: 12 companies
  ❌ Failed: 4 companies
```

## Troubleshooting

### Database Connection Error
```
❌ Database initialization failed: could not connect to server
```

**Solution**: Check your `.env` file has valid `DATABASE_URL` or `PG_VECTOR`

### Ticker Fetch Failed
```
❌ Failed to fetch S&P 500 tickers from all sources
```

**Solution**: Check internet connection. The script tries 3 sources (Wikipedia, SPY ETF, SlickCharts)

### Low Success Rate
```
⚠️  Low success rate (45.2%), please review errors
```

**Solution**: Review the error logs in `sp500_10k_ingestion.log` for specific failures

## Advanced Usage

### Run in background with logs
```bash
nohup python agent/rag/data_ingestion/ingest_sp500_10k.py --workers 5 > sp500_ingestion.out 2>&1 &

# Monitor progress
tail -f sp500_ingestion.out
tail -f sp500_10k_ingestion.log
```

### Batch processing (split into chunks)
```bash
# Process in batches of 100
for i in {0..4}; do
  skip=$((i * 100))
  python agent/rag/data_ingestion/ingest_sp500_10k.py \
    --skip-first $skip \
    --max-tickers 100 \
    --workers 5
done
```

## Memory Management

**The script includes aggressive memory management to prevent crashes:**

1. **Separate Process Per Ticker** (Python 3.11+)
   - Each ticker gets a completely fresh process
   - No memory accumulation between tickers
   - Uses `max_tasks_per_child=1` in ProcessPoolExecutor

2. **Aggressive Garbage Collection** (All Python versions)
   - Triple garbage collection after each ticker
   - Explicit cleanup of temporary files
   - Clears finqual and datamule references

3. **Automatic Cleanup**
   - Temporary files cleaned after each company
   - Database connections properly closed
   - Works even if ticker processing fails

**If you still experience memory issues:**
- Use fewer workers: `--workers 2` or `--workers 1`
- Process in smaller batches: `--max-tickers 50`
- Use sequential mode: `--sequential`
- Upgrade to Python 3.11 or newer

## Notes

- The S&P 500 list is fetched dynamically, so it always uses the current constituents
- Companies that don't have 10-K filings in the date range will be marked as "No data found"
- Failed companies are logged with error details for debugging
- Temporary files are automatically cleaned up after each company
- Safe to interrupt and resume using `--skip-first`
- **Memory is aggressively managed** to prevent finqual/datamule memory leaks

## Next Steps

After ingestion completes, the 10-K data will be available for:
- RAG queries through the StrataLens AI agent
- Financial analysis and comparison
- Historical trend analysis
- Semantic search across all S&P 500 10-K filings
