# StrataLens AI

An open-source equity research platform for analyzing US public equity market documents and data using AI agents with RAG.

**Hosted Platform:** [stratalens.ai](https://stratalens.ai)


## Features

### Earnings Transcript Analysis (Live)
- RAG-based chat over earnings calls (2022-2025)
- Natural language queries with citation support
- Real-time streaming responses

### Financial Data Screener (In Repository)
- Text-to-duckdb conversion for natural language queries
- Financial metrics, company screening and some qualitative metrics
- **Note:** Not optimized for production deployment

### Company Financial Information (Planned)
- Financial statements and key metrics
- Company profiles and search
- Had earlier developed with Financial modelling prep, but want to parse SEC filings to get company fundamentals this time.

### SEC 10-K Filings (In Development)
- RAG-based analysis of SEC filings
- Progress tracked in `experiments/` folder
- Benchmarking with an LLM as a judge on financebench dataset

## Technology Stack

- **Backend:** FastAPI, PostgreSQL with pgvector
- **AI/ML:** OpenAI GPT-4, Groq Llama, LangChain, Sentence Transformers
- **Infrastructure:** Redis, WebSockets

## Getting Started

### Prerequisites

- Python 3.9+
- PostgreSQL 12+ with pgvector extension
- Redis (optional, for caching)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stratalens_ai.git
cd stratalens_ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file with required variables:
```bash
DATABASE_URL=postgresql://user:password@host:port/database
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
REDIS_URL=redis://localhost:6379
JWT_SECRET_KEY=your_secret_key
```

4. Initialize database:
```bash
python utils/database_init.py
```

5. Run the application:
```bash
python fastapi_server.py
```

The API will be available at `http://localhost:8000`

## Documentation

### Data Ingestion Pipeline

The RAG data ingestion pipeline processes earnings transcripts for semantic search. See `agent/rag/data_ingestion/README.md` for setup instructions.

### API Documentation

Interactive API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key Endpoints

- **Chat & RAG:** `/message/stream-v2` - Conversational interface with streaming
- **Company Search:** `/companies/search` - Search by name or ticker
- **Financial Data:** `/screener/query/stream` - Natural language queries
- **Earnings Transcripts:** `/transcript/{ticker}/{year}/{quarter}` - Specific transcripts

## Development Status

### Production (Live on stratalens.ai)
- Earnings transcript analysis and search (2022-2025)
- Real-time streaming chat interface with RAG
- User authentication and management

### Available in Repository
- Company search and profile information
- Financial data access and screener (under optimization)

### Under Development
- SEC 10-K filings integration and analysis
- Financial screener performance optimization
- Additional data sources and integrations

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss proposed modifications.

## Support

For issues, feature requests, or questions, please open an issue in the GitHub repository. 