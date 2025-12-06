#!/usr/bin/env python3
"""
Comprehensive SEC Filings Data Loading Module

This module handles downloading, parsing, processing, and caching SEC filings data.
It creates embeddings and saves everything to disk for fast loading later.

Simply modify the variables at the top of this file to configure what to load.
"""

import logging
import numpy as np
import pickle
import os
import re
import tempfile
import datetime
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# External dependencies
import pandas as pd
from datasets import load_dataset
from datamule import Portfolio
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# CONFIGURATION - Modify these variables to change what gets loaded
# =============================================================================

# Company to load
TICKER = "ADBE"  # Company ticker symbol (e.g., "ADBE", "MSFT", "AAPL")

# Year configuration
FISCAL_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]  # All years for FinanceBench
# The system will automatically search for filings in the range [min_year-1, max_year+1]
# to account for cases where fiscal year doesn't match filing year

# FinanceBench companies (updated with actual company names from dataset)
FINANCEBENCH_COMPANIES = {
    "3M": "MMM",
    "AES Corporation": "AES",
    "AMD": "AMD",
    "Activision Blizzard": "ATVI",
    "Adobe": "ADBE",
    "Amazon": "AMZN",
    "Amcor": "AMCR",
    "American Express": "AXP",
    "American Water Works": "AWK",
    "Best Buy": "BBY",
    "Block": "SQ",
    "Boeing": "BA",
    "CVS Health": "CVS",
    "Coca-Cola": "KO",
    "Corning": "GLW",
    "Costco": "COST",
    "General Mills": "GIS",
    "JPMorgan": "JPM",
    "Johnson & Johnson": "JNJ",
    "Kraft Heinz": "KHC",
    "Lockheed Martin": "LMT",
    "MGM Resorts": "MGM",
    "Microsoft": "MSFT",
    "Netflix": "NFLX",
    "Nike": "NKE",
    "Paypal": "PYPL",
    "PepsiCo": "PEP",
    "Pfizer": "PFE",
    "Ulta Beauty": "ULTA",
    "Verizon": "VZ",
    "Walmart": "WMT"
}

# FinanceBench loading options
LOAD_FINANCEBENCH_COMPANIES = True  # Set to True to use load_all_financebench_companies (required for SPECIFIC_COMPANY_YEARS)
FINANCEBENCH_COMPANY_FILTER = None  # All FinanceBench companies
FINANCEBENCH_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]  # All years

# Specific company-year combinations to load (when USE_SPECIFIC_COMPANY_YEARS is True)
USE_SPECIFIC_COMPANY_YEARS = True  # Set to True to load specific company-year combinations
SPECIFIC_COMPANY_YEARS = {
    # Failed cases from FinanceBench evaluation
    "AES Corporation": [2022],
    "AMD": [2022],
    "Amcor": [2023],
    "CVS Health": [2018, 2022],
    "Coca-Cola": [2017],
    "General Mills": [2020, 2022],
    "Johnson & Johnson": [2022],
    "Netflix": [2015],
    "PepsiCo": [2022],
    "Pfizer": [2021],
    "Verizon": [2022],
}

# Example configurations:
# 
# For single company (current default):
# LOAD_FINANCEBENCH_COMPANIES = False
# TICKER = "ADBE"
# FISCAL_YEARS = [2020, 2021, 2022, 2023, 2024]
#
# For all FinanceBench companies:
# LOAD_FINANCEBENCH_COMPANIES = True
# FINANCEBENCH_COMPANY_FILTER = None
# FINANCEBENCH_YEARS = [2020, 2021, 2022, 2023, 2024]
#
# For specific FinanceBench company (e.g., Adobe):
# LOAD_FINANCEBENCH_COMPANIES = True
# FINANCEBENCH_COMPANY_FILTER = "Adobe"
# FINANCEBENCH_YEARS = [2020, 2021, 2022, 2023, 2024]

# Cache directory
CACHE_DIR = "embeddings_cache"

# Check cache status instead of loading
CHECK_CACHE_ONLY = False

# =============================================================================
# END CONFIGURATION
# =============================================================================

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# SEC 10-K Section definitions
SEC_10K_SECTIONS = {
    "item_1": {
        "title": "Business",
        "keywords": ["business", "operations", "products", "services", "company overview"]
    },
    "item_1a": {
        "title": "Risk Factors",
        "keywords": ["risk", "risks", "risk factors", "uncertainty", "challenges"]
    },
    "item_1b": {
        "title": "Unresolved Staff Comments",
        "keywords": ["staff comments", "unresolved", "sec comments"]
    },
    "item_2": {
        "title": "Properties",
        "keywords": ["properties", "facilities", "real estate", "locations"]
    },
    "item_3": {
        "title": "Legal Proceedings",
        "keywords": ["legal", "proceedings", "litigation", "lawsuits", "court"]
    },
    "item_5": {
        "title": "Market for Common Equity",
        "keywords": ["equity", "stock", "market", "shareholders", "common stock"]
    },
    "item_7": {
        "title": "Management's Discussion and Analysis (MD&A)",
        "keywords": ["md&a", "management discussion", "analysis", "performance", "results", "revenue", "sales"]
    },
    "item_7a": {
        "title": "Market Risk Disclosures",
        "keywords": ["market risk", "risk disclosure", "hedging", "derivatives"]
    },
    "item_8": {
        "title": "Financial Statements",
        "keywords": ["financial statements", "balance sheet", "income statement", "cash flow", "revenue", "expenses"]
    },
    "item_9a": {
        "title": "Controls and Procedures",
        "keywords": ["controls", "procedures", "internal controls", "compliance"]
    },
    "item_10": {
        "title": "Directors and Officers",
        "keywords": ["directors", "executives", "officers", "governance", "board"]
    },
    "item_11": {
        "title": "Executive Compensation",
        "keywords": ["compensation", "pay", "salaries", "benefits", "stock options", "executive"]
    },
    "item_12": {
        "title": "Security Ownership",
        "keywords": ["ownership", "beneficial owners", "shareholders", "insider"]
    },
    "item_15": {
        "title": "Exhibits and Schedules",
        "keywords": ["exhibits", "schedules", "attachments"]
    }
}

class DataProcessor:
    """
    Handles data processing and embeddings for SEC RAG system
    
    Responsibilities:
    - Chunk preparation from filing data
    - Embedding creation (semantic + TF-IDF)
    - Table processing utilities
    - Data transformation utilities
    """
    
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        """
        Initialize the data processor

        Args:
            embedding_model_name: Name of the sentence transformer model to use
        """
        logger.info(f"🔧 Initializing Data Processor...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        logger.info(f"📦 Loading cross-encoder for reranking...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Data storage
        self.chunks = []
        self.embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.tables = {}  # Store tables separately

        logger.info(f"✅ Data Processor initialized")
    
    def prepare_chunks(self, filing_data, use_hierarchical=True, exclude_titles=True):
        """
        Convert content to searchable chunks for RAG

        Args:
            filing_data: Dict from download_and_extract_10k
            use_hierarchical: If True, use contextual_chunks; if False, use old table method
            exclude_titles: If True, exclude title/heading chunks (use only text and tables)
        """
        ticker = filing_data.get('ticker', 'UNKNOWN')

        if use_hierarchical and 'contextual_chunks' in filing_data:
            logger.info("🔧 Preparing RAG chunks (text + tables, keeping hierarchy + section metadata)...")

            # Reset tables dictionary
            self.tables = {}
            table_index = 0

            for chunk in filing_data['contextual_chunks']:
                # Skip images and very short content
                if chunk['type'] == 'image' or len(chunk['content']) < 20:
                    continue

                # SKIP TITLES/HEADINGS - only use text and tables for RAG
                if exclude_titles and chunk['type'] in ['title', 'heading']:
                    continue

                # Use original content for all chunks (no context added)
                content_to_use = chunk['content']

                # If this is a table chunk, store it in the tables dictionary
                if chunk['type'] == 'table':
                    table_id = f"{ticker}_table_{table_index}"
                    self.tables[table_id] = {
                        'id': table_id,
                        'content': content_to_use,
                        'table_data': chunk.get('metadata', {}).get('table_data', None),
                        'path': chunk.get('path', []),
                        'path_string': chunk.get('path_string', ''),
                        'sec_section': chunk.get('sec_section', 'unknown'),
                        'sec_section_title': chunk.get('sec_section_title', 'Unknown')
                    }
                    table_index += 1

                self.chunks.append({
                    'id': chunk['id'],
                    'ticker': ticker,
                    'content': content_to_use,  # Original content only
                    'original_content': chunk.get('original_content', chunk['content']),
                    'path': chunk.get('path', []),
                    'path_string': chunk.get('path_string', ''),
                    'type': chunk['type'],
                    'level': chunk.get('level', 0),
                    'sec_section': chunk.get('sec_section', 'unknown'),  # Section ID
                    'sec_section_title': chunk.get('sec_section_title', 'Unknown'),  # Section name
                    'context_before': chunk.get('context_before', []),
                    'context_after': chunk.get('context_after', [])
                })

            logger.info(f"✅ Created {len(self.chunks)} RAG chunks")
            logger.info(f"✅ Extracted {len(self.tables)} tables to separate dictionary")

            # Show breakdown by type and section
            type_counts = {}
            section_counts = {}
            for chunk in self.chunks:
                t = chunk['type']
                s = chunk.get('sec_section', 'unknown')
                type_counts[t] = type_counts.get(t, 0) + 1
                section_counts[s] = section_counts.get(s, 0) + 1

            logger.info(f"   Type breakdown: {type_counts}")
            logger.info(f"   Section breakdown: {section_counts}")

            # FLAG: Check if zero table chunks were found
            table_count = type_counts.get('table', 0)
            if table_count == 0:
                logger.warning(f"⚠️  WARNING: ZERO TABLE CHUNKS FOUND FOR {ticker}!")
                logger.warning(f"⚠️  This may indicate an issue with table extraction or document structure")
        else:
            # Fallback to old table method
            logger.info("🔧 Preparing table chunks (legacy mode)...")
            tables = filing_data.get('tables', [])
            
            for idx, table in enumerate(tables):
                table_text = self._table_to_text(table)
                
                if table_text:
                    self.chunks.append({
                        'id': f"{ticker}_table_{idx}",
                        'ticker': ticker,
                        'table_index': idx,
                        'content': table_text,
                        'type': 'table',
                        'path': [],
                        'path_string': '',
                        'sec_section': 'unknown',
                        'sec_section_title': 'Unknown'
                    })
            
            logger.info(f"✅ Created {len(self.chunks)} table chunks")
    
    def _table_to_text(self, table):
        """Convert table to text"""
        if isinstance(table, str):
            return table
        elif isinstance(table, dict):
            return " | ".join(f"{k}: {v}" for k, v in table.items())
        elif isinstance(table, list):
            return " | ".join(str(item) for item in table)
        return str(table)
    
    def create_embeddings(self):
        """Create embeddings and TF-IDF vectors for all chunks"""
        logger.info(f"🧠 Creating embeddings for {len(self.chunks)} chunks...")
        
        texts = [chunk['content'] for chunk in self.chunks]
        
        # Create semantic embeddings
        self.embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        logger.info(f"✅ Embeddings created: {self.embeddings.shape}")
        
        # Create TF-IDF vectors for keyword search
        logger.info(f"🔤 Creating TF-IDF vectors for keyword search...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        logger.info(f"✅ TF-IDF vectors created: {self.tfidf_matrix.shape}")
    
    def get_chunks(self):
        """Get the processed chunks"""
        return self.chunks
    
    def get_embeddings(self):
        """Get the semantic embeddings"""
        return self.embeddings
    
    def get_tfidf_vectorizer(self):
        """Get the TF-IDF vectorizer"""
        return self.tfidf_vectorizer
    
    def get_tfidf_matrix(self):
        """Get the TF-IDF matrix"""
        return self.tfidf_matrix
    
    def get_cross_encoder(self):
        """Get the cross-encoder for reranking"""
        return self.cross_encoder
    
    def get_embedding_model(self):
        """Get the embedding model"""
        return self.embedding_model

    def get_tables(self):
        """Get the tables dictionary"""
        return self.tables

    def get_financial_tables_index(self):
        """Get financial tables index"""
        return self.financial_tables_index if hasattr(self, 'financial_tables_index') else {}

    def has_embeddings(self):
        """Check if embeddings are available"""
        return self.embeddings is not None and self.tfidf_matrix is not None
    
    
    def identify_financial_statement_tables(self):
        """
        Identify and label the three core financial statement tables.
        Uses both path and content matching for robustness.
        Also uses structural indicators for tables without standard titles.
        Returns metadata about which tables are which.
        """
        financial_tables_index = {
            'income_statement': None,
            'balance_sheet': None,
            'cash_flow': None
        }

        logger.info("🔍 Identifying core financial statement tables...")

        for table_id, table_data in self.tables.items():
            path = table_data.get('path', [])
            path_str = ' > '.join(str(p) for p in path).lower()

            # Get table content for matching (first 1500 chars for better structural matching)
            content = table_data.get('content', '').lower()
            content_preview = content[:1500]

            # Combined search string (path + content for better matching)
            search_str = path_str + ' ' + content_preview

            # Skip if this is a table of contents (page number references)
            if '| page' in content_preview[:200] and 'consolidated statement' in content_preview[:200]:
                table_data['is_financial_statement'] = False
                table_data['statement_type'] = None
                table_data['priority'] = 'NORMAL'
                continue

            # Income Statement - check titles AND structural indicators
            if any(kw in search_str for kw in [
                'consolidated statements of operations',
                'consolidated statement of operations',
                'consolidated statements of income',
                'consolidated statement of income',
                'consolidated statements of earnings',
                'consolidated statement of earnings'
            ]):
                table_data['is_financial_statement'] = True
                table_data['statement_type'] = 'income_statement'
                table_data['priority'] = 'CRITICAL'
                if financial_tables_index['income_statement'] is None:
                    financial_tables_index['income_statement'] = table_id
                    logger.info(f"   ✅ Income Statement: {table_id}")

            # Income Statement - structural matching (for tables without titles)
            elif (financial_tables_index['income_statement'] is None and
                  all(kw in content_preview for kw in ['operating income', 'net income']) and
                  any(kw in content_preview for kw in ['revenue', 'gross profit', 'operating revenue'])):
                table_data['is_financial_statement'] = True
                table_data['statement_type'] = 'income_statement'
                table_data['priority'] = 'CRITICAL'
                financial_tables_index['income_statement'] = table_id
                logger.info(f"   ✅ Income Statement (structural): {table_id}")

            # Balance Sheet - check titles AND structural indicators
            elif any(kw in search_str for kw in [
                'consolidated balance sheets',
                'consolidated balance sheet',
                'consolidated statements of financial position'
            ]):
                table_data['is_financial_statement'] = True
                table_data['statement_type'] = 'balance_sheet'
                table_data['priority'] = 'CRITICAL'
                if financial_tables_index['balance_sheet'] is None:
                    financial_tables_index['balance_sheet'] = table_id
                    logger.info(f"   ✅ Balance Sheet: {table_id}")

            # Balance Sheet - structural matching (for tables without titles)
            elif (financial_tables_index['balance_sheet'] is None and
                  'assets' in content_preview and
                  'current assets' in content_preview and
                  ('total assets' in content_preview or 'total current assets' in content_preview) and
                  ('liabilities' in content_preview or 'equity' in content_preview)):
                table_data['is_financial_statement'] = True
                table_data['statement_type'] = 'balance_sheet'
                table_data['priority'] = 'CRITICAL'
                financial_tables_index['balance_sheet'] = table_id
                logger.info(f"   ✅ Balance Sheet (structural): {table_id}")

            # Cash Flow - check titles AND structural indicators
            elif any(kw in search_str for kw in [
                'consolidated statements of cash flows',
                'consolidated statement of cash flows'
            ]):
                table_data['is_financial_statement'] = True
                table_data['statement_type'] = 'cash_flow'
                table_data['priority'] = 'CRITICAL'
                if financial_tables_index['cash_flow'] is None:
                    financial_tables_index['cash_flow'] = table_id
                    logger.info(f"   ✅ Cash Flow: {table_id}")

            # Cash Flow - structural matching (for tables without titles)
            elif (financial_tables_index['cash_flow'] is None and
                  'operating activities' in content_preview and
                  ('investing activities' in content_preview or 'financing activities' in content_preview) and
                  'cash' in content_preview):
                table_data['is_financial_statement'] = True
                table_data['statement_type'] = 'cash_flow'
                table_data['priority'] = 'CRITICAL'
                financial_tables_index['cash_flow'] = table_id
                logger.info(f"   ✅ Cash Flow (structural): {table_id}")

            else:
                table_data['is_financial_statement'] = False
                table_data['statement_type'] = None
                table_data['priority'] = 'NORMAL'

        # Log summary
        found_count = sum(1 for v in financial_tables_index.values() if v is not None)
        logger.info(f"📊 Found {found_count}/3 core financial statement tables")

        return financial_tables_index

    def save_embeddings_to_disk(self, ticker, year, base_dir="embeddings_cache", hierarchical_data=None):
        """
        Save embeddings and related data to disk for future reuse.

        Args:
            ticker: Company ticker symbol
            year: Fiscal year
            base_dir: Base directory for saving embeddings
            hierarchical_data: Optional hierarchical chunks and document data to save

        Returns:
            Path to the saved data directory
        """
        if not self.has_embeddings():
            logger.warning("⚠️  No embeddings to save")
            return None

        # Create directory structure: embeddings_cache/TICKER/YEAR/
        save_dir = Path(base_dir) / ticker / str(year)
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"💾 Saving embeddings and hierarchical data to {save_dir}...")

        # Identify and label financial statement tables BEFORE saving
        financial_tables_index = self.identify_financial_statement_tables()

        try:
            # Save processed chunks (for embeddings)
            with open(save_dir / "chunks.pkl", "wb") as f:
                pickle.dump(self.chunks, f)
            
            # Save embeddings (numpy array)
            np.save(save_dir / "embeddings.npy", self.embeddings)
            
            # Save TF-IDF vectorizer and matrix
            with open(save_dir / "tfidf_vectorizer.pkl", "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)
            
            # Save TF-IDF matrix (scipy sparse matrix)
            with open(save_dir / "tfidf_matrix.pkl", "wb") as f:
                pickle.dump(self.tfidf_matrix, f)
            
            # Save tables dictionary (always save, even if empty)
            with open(save_dir / "tables.pkl", "wb") as f:
                pickle.dump(self.tables, f)
            logger.info(f"✅ Saved {len(self.tables)} tables to tables.pkl")

            # Save financial tables index for quick access
            import json
            with open(save_dir / "financial_tables_index.json", "w") as f:
                json.dump(financial_tables_index, f, indent=2)
            logger.info(f"📊 Saved financial tables index")

            # Save hierarchical data if provided
            if hierarchical_data:
                logger.info(f"📁 Saving hierarchical data...")

                # Save hierarchical chunks (raw extracted chunks)
                if 'hierarchical_chunks' in hierarchical_data:
                    with open(save_dir / "hierarchical_chunks.pkl", "wb") as f:
                        pickle.dump(hierarchical_data['hierarchical_chunks'], f)

                # Save contextual chunks (processed hierarchical chunks)
                if 'contextual_chunks' in hierarchical_data:
                    with open(save_dir / "contextual_chunks.pkl", "wb") as f:
                        pickle.dump(hierarchical_data['contextual_chunks'], f)

                # Save document text
                if 'document_text' in hierarchical_data:
                    with open(save_dir / "document_text.txt", "w", encoding="utf-8") as f:
                        f.write(hierarchical_data['document_text'])

                # Save document metadata
                doc_metadata = {
                    'document_length': hierarchical_data.get('document_length', 0),
                    'text_preview': hierarchical_data.get('text_preview', ''),
                    'num_hierarchical_chunks': len(hierarchical_data.get('hierarchical_chunks', [])),
                    'num_contextual_chunks': len(hierarchical_data.get('contextual_chunks', []))
                }

                with open(save_dir / "document_metadata.pkl", "wb") as f:
                    pickle.dump(doc_metadata, f)

                logger.info(f"✅ Saved hierarchical data: {doc_metadata['num_hierarchical_chunks']} hierarchical chunks, {doc_metadata['num_contextual_chunks']} contextual chunks")
            
            # Save metadata
            metadata = {
                'ticker': ticker,
                'year': year,
                'num_chunks': len(self.chunks),
                'embedding_shape': self.embeddings.shape,
                'tfidf_shape': self.tfidf_matrix.shape,
                'embedding_model_name': self.embedding_model.get_sentence_embedding_dimension(),
                'created_at': datetime.now().isoformat(),
                'chunk_types': self._get_chunk_type_breakdown(),
                'sections': self._get_section_breakdown(),
                'has_hierarchical_data': hierarchical_data is not None
            }
            
            with open(save_dir / "metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)
            
            logger.info(f"✅ Successfully saved embeddings to {save_dir}")
            logger.info(f"   - {len(self.chunks)} processed chunks")
            logger.info(f"   - {self.embeddings.shape} embeddings")
            logger.info(f"   - {self.tfidf_matrix.shape} TF-IDF matrix")
            
            return save_dir
            
        except Exception as e:
            logger.error(f"❌ Failed to save embeddings: {e}")
            return None
    
    def load_embeddings_from_disk(self, ticker, year, base_dir="embeddings_cache"):
        """
        Load embeddings and related data from disk.
        
        Args:
            ticker: Company ticker symbol
            year: Fiscal year
            base_dir: Base directory for loading embeddings
            
        Returns:
            True if successful, False otherwise
        """
        load_dir = Path(base_dir) / ticker / str(year)
        
        if not load_dir.exists():
            logger.warning(f"⚠️  Embeddings directory not found: {load_dir}")
            return False
        
        logger.info(f"📥 Loading embeddings from {load_dir}...")
        
        try:
            # Load chunks
            with open(load_dir / "chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)
            
            # Load embeddings
            self.embeddings = np.load(load_dir / "embeddings.npy")
            
            # Load TF-IDF vectorizer and matrix
            with open(load_dir / "tfidf_vectorizer.pkl", "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            with open(load_dir / "tfidf_matrix.pkl", "rb") as f:
                self.tfidf_matrix = pickle.load(f)

            # Load tables dictionary
            tables_file = load_dir / "tables.pkl"
            if tables_file.exists():
                with open(tables_file, "rb") as f:
                    self.tables = pickle.load(f)
                logger.info(f"✅ Loaded {len(self.tables)} tables")
            else:
                self.tables = {}
                logger.warning(f"⚠️  No tables.pkl file found - tables will be empty")

            # Load financial tables index
            financial_index_file = load_dir / "financial_tables_index.json"
            if financial_index_file.exists():
                with open(financial_index_file, "r") as f:
                    self.financial_tables_index = json.load(f)
                found_count = sum(1 for v in self.financial_tables_index.values() if v is not None)
                logger.info(f"📊 Loaded financial tables index ({found_count}/3 found)")
            else:
                # Auto-generate the financial index if it doesn't exist
                logger.info(f"📊 No financial tables index found - generating now...")
                self.financial_tables_index = self.identify_financial_statement_tables()

                # Save it immediately for future use
                try:
                    with open(financial_index_file, "w") as f:
                        json.dump(self.financial_tables_index, f, indent=2)

                    # Also re-save tables.pkl with updated metadata
                    with open(load_dir / "tables.pkl", "wb") as f:
                        pickle.dump(self.tables, f)

                    found_count = sum(1 for v in self.financial_tables_index.values() if v is not None)
                    logger.info(f"📊 Generated and saved financial tables index ({found_count}/3 found)")
                except Exception as e:
                    logger.warning(f"⚠️  Could not save generated index: {e}")

            # Load metadata
            with open(load_dir / "metadata.pkl", "rb") as f:
                metadata = pickle.load(f)

            logger.info(f"✅ Successfully loaded embeddings from {load_dir}")
            logger.info(f"   - {len(self.chunks)} chunks")
            logger.info(f"   - {self.embeddings.shape} embeddings")
            logger.info(f"   - {self.tfidf_matrix.shape} TF-IDF matrix")
            logger.info(f"   - {len(self.tables)} tables")
            logger.info(f"   - Created: {metadata.get('created_at', 'Unknown')}")

            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")
            return False
    
    def load_hierarchical_data_from_disk(self, ticker, year, base_dir="embeddings_cache"):
        """
        Load hierarchical data from disk.
        
        Args:
            ticker: Company ticker symbol
            year: Fiscal year
            base_dir: Base directory for loading data
            
        Returns:
            Dictionary with hierarchical data or None if not found
        """
        load_dir = Path(base_dir) / ticker / str(year)
        
        if not load_dir.exists():
            logger.warning(f"⚠️  Hierarchical data directory not found: {load_dir}")
            return None
        
        logger.info(f"📥 Loading hierarchical data from {load_dir}...")
        
        try:
            hierarchical_data = {}
            
            # Load hierarchical chunks
            hierarchical_chunks_file = load_dir / "hierarchical_chunks.pkl"
            if hierarchical_chunks_file.exists():
                with open(hierarchical_chunks_file, "rb") as f:
                    hierarchical_data['hierarchical_chunks'] = pickle.load(f)
                logger.info(f"✅ Loaded {len(hierarchical_data['hierarchical_chunks'])} hierarchical chunks")
            
            # Load contextual chunks
            contextual_chunks_file = load_dir / "contextual_chunks.pkl"
            if contextual_chunks_file.exists():
                with open(contextual_chunks_file, "rb") as f:
                    hierarchical_data['contextual_chunks'] = pickle.load(f)
                logger.info(f"✅ Loaded {len(hierarchical_data['contextual_chunks'])} contextual chunks")
            
            # Load document text
            document_text_file = load_dir / "document_text.txt"
            if document_text_file.exists():
                with open(document_text_file, "r", encoding="utf-8") as f:
                    hierarchical_data['document_text'] = f.read()
                logger.info(f"✅ Loaded document text ({len(hierarchical_data['document_text'])} characters)")
            
            # Load document metadata
            doc_metadata_file = load_dir / "document_metadata.pkl"
            if doc_metadata_file.exists():
                with open(doc_metadata_file, "rb") as f:
                    doc_metadata = pickle.load(f)
                hierarchical_data.update(doc_metadata)
                logger.info(f"✅ Loaded document metadata")

            # Load tables
            tables_file = load_dir / "tables.pkl"
            if tables_file.exists():
                with open(tables_file, "rb") as f:
                    hierarchical_data['tables'] = pickle.load(f)
                logger.info(f"✅ Loaded {len(hierarchical_data['tables'])} tables")
            else:
                hierarchical_data['tables'] = {}
                logger.warning(f"⚠️  No tables.pkl file found - tables will be empty")

            if hierarchical_data:
                logger.info(f"✅ Successfully loaded hierarchical data for {ticker} {year}")
                return hierarchical_data
            else:
                logger.warning(f"⚠️  No hierarchical data found for {ticker} {year}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load hierarchical data: {e}")
            return None
    
    def embeddings_exist_on_disk(self, ticker, year, base_dir="embeddings_cache"):
        """
        Check if embeddings exist on disk for given ticker and year.
        
        Args:
            ticker: Company ticker symbol
            year: Fiscal year
            base_dir: Base directory for checking embeddings
            
        Returns:
            True if embeddings exist, False otherwise
        """
        load_dir = Path(base_dir) / ticker / str(year)
        required_files = ["chunks.pkl", "embeddings.npy", "tfidf_vectorizer.pkl", "tfidf_matrix.pkl", "metadata.pkl"]
        
        if not load_dir.exists():
            return False
        
        return all((load_dir / file).exists() for file in required_files)
    
    def hierarchical_data_exist_on_disk(self, ticker, year, base_dir="embeddings_cache"):
        """
        Check if hierarchical data exists on disk for given ticker and year.
        
        Args:
            ticker: Company ticker symbol
            year: Fiscal year
            base_dir: Base directory for checking data
            
        Returns:
            True if hierarchical data exists, False otherwise
        """
        load_dir = Path(base_dir) / ticker / str(year)
        required_files = ["hierarchical_chunks.pkl", "contextual_chunks.pkl", "document_text.txt", "document_metadata.pkl"]
        
        if not load_dir.exists():
            return False
        
        return all((load_dir / file).exists() for file in required_files)
    
    def _get_chunk_type_breakdown(self):
        """Get breakdown of chunk types for metadata"""
        type_counts = {}
        for chunk in self.chunks:
            chunk_type = chunk.get('type', 'unknown')
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        return type_counts
    
    def _get_section_breakdown(self):
        """Get breakdown of sections for metadata"""
        section_counts = {}
        for chunk in self.chunks:
            section = chunk.get('sec_section', 'unknown')
            section_counts[section] = section_counts.get(section, 0) + 1
        return section_counts

def extract_hierarchical_content(data, path=None, level=0, max_level=10):
    """
    Extract content from document data while preserving hierarchical structure.
    
    Args:
        data: Document data (dict/list/str)
        path: Current hierarchical path (e.g., ['Part I', 'Item 1', 'Business'])
        level: Current nesting level
        max_level: Maximum depth to traverse
    
    Returns:
        List of chunks, each containing:
        - content: The text content
        - path: Hierarchical path (list of section names)
        - type: Content type (title, text, table, etc.)
        - level: Nesting level
    """
    if path is None:
        path = []
    
    if level > max_level:
        return []
    
    chunks = []

    # Handle dictionary (most common case)
    if isinstance(data, dict):
        for key, value in data.items():
            # Skip metadata keys
            if key in ['metadata', 'parser', 'github', 'version']:
                continue
            
            current_path = path.copy()
            
            # Handle different content types
            if isinstance(value, dict):
                # Extract title if present
                title = value.get('title', '').strip()
                class_type = value.get('class', '')
                
                # Update path with title
                if title:
                    # Clean up title (remove checkboxes, extra spaces)
                    clean_title = re.sub(r'☒|☐|\xa0+', '', title).strip()
                    if clean_title and len(clean_title) > 2:
                        current_path.append(clean_title)
                
                # Extract text content
                if 'textsmall' in value or 'text' in value:
                    text_content = value.get('textsmall') or value.get('text', '')
                    if text_content and text_content.strip():
                        chunks.append({
                            'content': text_content.strip(),
                            'path': current_path.copy(),
                            'type': 'text',
                            'level': level,
                            'class': class_type
                        })
                
                # Extract title as separate chunk if it's significant
                if title:
                    clean_title = re.sub(r'☒|☐|\xa0+', '', title).strip()
                    if clean_title and len(clean_title) > 2 and 'header' in class_type.lower():
                        chunks.append({
                            'content': clean_title,
                            'path': current_path.copy(),
                            'type': 'title',
                            'level': level,
                            'class': class_type
                        })
                
                # Extract tables
                if 'table' in value and value['table']:
                    table_data = value['table']
                    if table_data:  # Not empty list
                        # Convert table to readable text
                        table_text = _format_table(table_data)
                        if table_text:
                            chunks.append({
                                'content': table_text,
                                'path': current_path.copy(),
                                'type': 'table',
                                'level': level,
                                'table_data': table_data
                            })
                
                # Extract images (metadata only)
                if 'image' in value and value['image']:
                    img_data = value['image']
                    chunks.append({
                        'content': f"Image: {img_data.get('alt', 'No description')}",
                        'path': current_path.copy(),
                        'type': 'image',
                        'level': level,
                        'image_src': img_data.get('src', '')
                    })
                
                # Recursively process 'contents' if present
                if 'contents' in value:
                    sub_chunks = extract_hierarchical_content(
                        value['contents'],
                        current_path,
                        level + 1,
                        max_level
                    )
                    chunks.extend(sub_chunks)
                
                # If this dict doesn't have recognized fields but has other nested data,
                # process it recursively (e.g., container dicts with numeric keys)
                elif not any(k in value for k in ['title', 'textsmall', 'text', 'table', 'image', 'class']):
                    # This is likely a container dict, recurse into it
                    sub_chunks = extract_hierarchical_content(
                        value,
                        current_path,
                        level,
                        max_level
                    )
                    chunks.extend(sub_chunks)
            
            elif isinstance(value, str):
                # Direct string value
                if value.strip():
                    chunks.append({
                        'content': value.strip(),
                        'path': current_path.copy(),
                        'type': 'text',
                        'level': level
                    })
            
            elif isinstance(value, list):
                # List of items
                sub_chunks = extract_hierarchical_content(
                    value,
                    current_path,
                    level,
                    max_level
                )
                chunks.extend(sub_chunks)
    
    # Handle list
    elif isinstance(data, list):
        for item in data:
            sub_chunks = extract_hierarchical_content(
                item,
                path,
                level,
                max_level
            )
            chunks.extend(sub_chunks)
    
    # Handle string
    elif isinstance(data, str):
        if data.strip():
            chunks.append({
                'content': data.strip(),
                'path': path.copy(),
                'type': 'text',
                'level': level
            })
    
    return chunks

def extract_text(data):
    """Simple text extraction (backward compatibility)"""
    # Handle both wrapped and unwrapped document structures
    doc_data = data.get('document', data) if isinstance(data, dict) and 'document' in data else data
    chunks = extract_hierarchical_content(doc_data)
    return ' '.join(chunk['content'] for chunk in chunks if chunk.get('content'))

def identify_section_from_path(path_list):
    """
    Identify SEC section from hierarchical path.
    
    Args:
        path_list: List of path elements (e.g., ['Part I', 'Item 1', 'Business'])
    
    Returns:
        String like 'item_1', 'item_7a', etc. or 'unknown'
    """
    if not path_list:
        return 'unknown'
    
    # Convert path to lowercase string
    path_string = " > ".join(path_list).lower()
    
    # Check each SEC section
    for section_key, section_data in SEC_10K_SECTIONS.items():
        # Extract item number (e.g., "1", "7a")
        item_num = section_key.replace('item_', '')
        
        # Check if path contains "item X" or section keywords
        if f"item {item_num}" in path_string or f"item{item_num}" in path_string:
            return section_key
        
        # Check keywords
        for keyword in section_data['keywords']:
            if keyword in path_string:
                return section_key
    
    return 'unknown'

def _format_table(table_data):
    """Convert table data to readable text format"""
    if not table_data:
        return ""
    
    if isinstance(table_data, list) and len(table_data) > 0:
        # Assume first row is header
        if isinstance(table_data[0], list):
            rows = []
            for row in table_data:
                rows.append(" | ".join(str(cell) for cell in row))
            return "\n".join(rows)
        else:
            return str(table_data)
    
    return str(table_data)

def create_contextual_chunks(hierarchical_chunks, window_size=3, combine_short=True, min_chunk_size=100):
    """
    Create chunks from hierarchical content with metadata (context tracking for reference only).
    
    Args:
        hierarchical_chunks: Output from extract_hierarchical_content
        window_size: Number of surrounding chunks to track for metadata
        combine_short: (Deprecated - not used)
        min_chunk_size: (Deprecated - not used)
    
    Returns:
        List of chunks with hierarchical metadata and SEC section identification
    """
    contextual_chunks = []
    
    for i, chunk in enumerate(hierarchical_chunks):
        # Build context string from path
        context_path = " > ".join(chunk['path']) if chunk['path'] else "Document Root"
        
        # Identify SEC section from path
        sec_section = identify_section_from_path(chunk['path'])
        sec_section_title = SEC_10K_SECTIONS.get(sec_section, {}).get('title', 'Unknown')
        
        # Get surrounding context (for better RAG)
        context_before = []
        context_after = []
        
        # Previous chunks (up to window_size)
        for j in range(max(0, i - window_size), i):
            if hierarchical_chunks[j]['type'] in ['text', 'title']:
                context_before.append(hierarchical_chunks[j]['content'])
        
        # Next chunks (up to window_size)
        for j in range(i + 1, min(len(hierarchical_chunks), i + 1 + window_size)):
            if hierarchical_chunks[j]['type'] in ['text', 'title']:
                context_after.append(hierarchical_chunks[j]['content'])
        
        # Build full chunk - use original content as-is (no surrounding context)
        full_content = chunk['content']
        
        contextual_chunks.append({
            'id': f"chunk_{i}",
            'content': full_content,
            'original_content': chunk['content'],
            'path': chunk['path'],
            'path_string': context_path,
            'type': chunk['type'],
            'level': chunk['level'],
            'sec_section': sec_section,  # NEW: SEC section identifier
            'sec_section_title': sec_section_title,  # NEW: Human-readable section name
            'context_before': context_before,
            'context_after': context_after,
            'metadata': {
                k: v for k, v in chunk.items() 
                if k not in ['content', 'path', 'type', 'level']
            }
        })
    
    return contextual_chunks

def extract_fiscal_year(data, path=None, is_root_call=True):
    """
    Extract fiscal year from SEC document data.
    
    Args:
        data: Document data (dict/list/str)
        path: Current hierarchical path (for debugging)
        is_root_call: Whether this is the root call (for logging control)
    
    Returns:
        String representing the fiscal year (e.g., "2024") or None if not found
        For non-root calls, returns a list of years found
    """
    if path is None:
        path = []
    
    year_references = []
    
    # Common patterns for fiscal year in SEC filings
    fiscal_year_patterns = [
        (r'fiscal year ended.*?(\d{4})', 'fiscal_year_ended'),
        (r'year ended.*?(\d{4})', 'year_ended'),
        (r'for the.*?year.*?(\d{4})', 'for_the_year'),
        (r'annual report.*?(\d{4})', 'annual_report'),
        (r'period ended.*?(\d{4})', 'period_ended'),
        (r'year.*?(\d{4}).*?ended', 'year_ended_reverse'),
        (r'(\d{4}).*?fiscal year', 'fiscal_year_prefix'),
        (r'(\d{4}).*?annual report', 'annual_report_prefix'),
        (r'fiscal.*?(\d{4})', 'fiscal_general'),
        (r'calendar year.*?(\d{4})', 'calendar_year'),
        (r'(\d{4}).*?calendar year', 'calendar_year_prefix'),
        (r'(\d{4}).*?fiscal', 'fiscal_prefix'),
        (r'(\d{4}).*?period', 'period_prefix'),
        (r'(\d{4}).*?ended', 'ended_prefix'),
        (r'(\d{4}).*?report', 'report_prefix'),
        (r'(\d{4}).*?filing', 'filing_prefix'),
        (r'filing.*?(\d{4})', 'filing_general'),
        (r'(\d{4}).*?form 10-k', 'form_10k_prefix'),
        (r'form 10-k.*?(\d{4})', 'form_10k_general'),
        # Add simple year patterns as fallback
        (r'\b(20\d{2})\b', 'simple_year_20xx'),
        (r'\b(19\d{2})\b', 'simple_year_19xx'),
    ]
    
    # Handle dictionary
    if isinstance(data, dict):
        for key, value in data.items():
            # Skip metadata keys
            if key in ['metadata', 'parser', 'github', 'version']:
                continue
            
            # Check text content for fiscal year patterns
            if isinstance(value, dict):
                # Check text fields
                for text_field in ['text', 'textsmall', 'title']:
                    if text_field in value and value[text_field]:
                        text_content = str(value[text_field])
                        text_lower = text_content.lower()
                        
                        for pattern, pattern_type in fiscal_year_patterns:
                            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                            for match in matches:
                                year = match.group(1)
                                # Validate year (reasonable range for SEC filings)
                                if 1990 <= int(year) <= 2030:
                                    year_references.append(year)
                
                # Recursively search in contents
                if 'contents' in value:
                    nested_years = extract_fiscal_year(value['contents'], path + [key], is_root_call=False)
                    if nested_years:
                        year_references.extend(nested_years)
                
                # Recursively search in other nested data
                elif not any(k in value for k in ['title', 'textsmall', 'text', 'table', 'image', 'class']):
                    nested_years = extract_fiscal_year(value, path + [key], is_root_call=False)
                    if nested_years:
                        year_references.extend(nested_years)
            
            elif isinstance(value, str):
                text_content = str(value)
                text_lower = text_content.lower()
                
                for pattern, pattern_type in fiscal_year_patterns:
                    matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                    for match in matches:
                        year = match.group(1)
                        if 1990 <= int(year) <= 2030:
                            year_references.append(year)
            
            elif isinstance(value, list):
                nested_years = extract_fiscal_year(value, path + [key], is_root_call=False)
                if nested_years:
                    year_references.extend(nested_years)
    
    # Handle list
    elif isinstance(data, list):
        for i, item in enumerate(data):
            nested_years = extract_fiscal_year(item, path + [str(i)], is_root_call=False)
            if nested_years:
                year_references.extend(nested_years)
    
    # Handle string
    elif isinstance(data, str):
        text_content = str(data)
        text_lower = text_content.lower()
        
        for pattern, pattern_type in fiscal_year_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                year = match.group(1)
                if 1990 <= int(year) <= 2030:
                    year_references.append(year)
    
    # Return the most common year found
    if year_references:
        year_counts = {}
        for year in year_references:
            year_counts[year] = year_counts.get(year, 0) + 1
        
        # Only log for root call to avoid spam
        if is_root_call:
            logger.info(f"🔍 Found year references: {year_counts}")
            logger.info(f"📊 Total year references found: {len(year_references)}")
        
        # Return the year with highest count
        selected_year = max(year_counts.items(), key=lambda x: x[1])[0]
        if is_root_call:
            logger.info(f"🎯 Selected fiscal year: {selected_year} (appeared {year_counts[selected_year]} times)")
            return selected_year
        else:
            # For non-root calls, return the list of years
            return year_references
    
    if is_root_call:
        logger.info("❌ No year references found in document")
        return None
    else:
        # For non-root calls, return empty list
        return []

def calculate_filing_year_range(fiscal_years):
    """
    Calculate the filing year range based on requested fiscal years.
    Adds buffer years to account for cases where fiscal year doesn't match filing year.
    
    Args:
        fiscal_years: List of fiscal years to load
        
    Returns:
        Tuple of (start_year, end_year) for filing search
    """
    if not fiscal_years:
        return None, None
    
    min_fiscal = min(fiscal_years)
    max_fiscal = max(fiscal_years)
    
    # Add buffer years: fiscal year can be filed in the following year
    # e.g., fiscal year 2023 might be filed in 2024
    start_year = min_fiscal - 1  # Look back one year
    end_year = max_fiscal + 1    # Look forward one year
    
    return start_year, end_year

def download_and_extract_10k(ticker, start_year, end_year):
    """
    Download and extract 10-K for a given ticker and year range.
    Returns a dictionary organized by fiscal year.
    """
    portfolio = Portfolio(ticker)
    portfolio.download_submissions(
        ticker=ticker,
        filing_date=(f'{start_year}-01-01', f'{end_year}-12-31'),
        submission_type=['10-K'])

    # Dictionary to store data organized by fiscal year
    filings_by_year = {}
    
    for document in portfolio.document_type('10-K'):
        
        document.parse()
        logger.info(f"🌳 Parsing hierarchical structure...")
        
        # Handle both wrapped and unwrapped document structures
        doc_data = document.data.get('document', document.data) if isinstance(document.data, dict) else document.data
        
        # Extract fiscal year first to use as key
        fiscal_year = extract_fiscal_year(document.data)
        if not fiscal_year:
            logger.warning(f"⚠️  Could not extract fiscal year, skipping document")
            continue
        
        # Convert fiscal year to integer for consistent comparison
        fiscal_year = int(fiscal_year)
            
        logger.info(f"📅 Processing fiscal year: {fiscal_year} (extracted from document content)")
        logger.info(f"📅 Filing year range: {start_year}-{end_year} (used for document search)")
        
        # Extract hierarchical content
        hierarchical_chunks = extract_hierarchical_content(doc_data)
        logger.info(f"📦 Extracted {len(hierarchical_chunks)} hierarchical chunks")

        # Show sample chunks
        if hierarchical_chunks:
            logger.info(f"📝 Sample chunks:")
            for i, chunk in enumerate(hierarchical_chunks[:3], 1):
                content_preview = chunk['content'][:100].replace('\n', ' ')
                logger.info(f"   Chunk {i} ({chunk['type']}): {content_preview}...")
                if chunk.get('path'):
                    logger.info(f"      Path: {' > '.join(chunk['path'][:3])}")
        
        # Extract document text
        document_text = extract_text(document.data)
        logger.info(f"📄 Document length: {len(document_text)} characters")
        
        # Show text preview
        text_preview = document_text[:300].replace('\n', ' ') if document_text else ""
        logger.info(f"   Text preview: {text_preview}...")
        
        # Create contextual chunks for RAG
        logger.info(f"🔗 Creating contextual chunks...")
        contextual_chunks = create_contextual_chunks(hierarchical_chunks)
        logger.info(f"✅ Created {len(contextual_chunks)} contextual chunks")

        # Count tables in this document
        table_count = sum(1 for chunk in hierarchical_chunks if chunk.get('type') == 'table')

        # Store data organized by fiscal year
        # If we already have data for this fiscal year, keep the one with more tables
        if fiscal_year in filings_by_year:
            existing_table_count = filings_by_year[fiscal_year].get('_table_count', 0)
            if table_count > existing_table_count:
                logger.info(f"📊 Replacing previous data for fiscal year {fiscal_year} ({existing_table_count} tables) with new data ({table_count} tables)")
                filings_by_year[fiscal_year] = {
                    "hierarchical_chunks": hierarchical_chunks,
                    "contextual_chunks": contextual_chunks,
                    "document_text": document_text,
                    "text_preview": text_preview,
                    "document_length": len(document_text),
                    "_table_count": table_count  # Internal field for tracking
                }
            else:
                logger.info(f"📊 Keeping previous data for fiscal year {fiscal_year} ({existing_table_count} tables) over new data ({table_count} tables)")
        else:
            filings_by_year[fiscal_year] = {
                "hierarchical_chunks": hierarchical_chunks,
                "contextual_chunks": contextual_chunks,
                "document_text": document_text,
                "text_preview": text_preview,
                "document_length": len(document_text),
                "_table_count": table_count  # Internal field for tracking
            }
            logger.info(f"✅ Stored data for fiscal year {fiscal_year} ({table_count} tables)")

    return filings_by_year


def create_embeddings_for_filing_data(filing_data, ticker, year, use_hierarchical=True, exclude_titles=True, cache_dir="embeddings_cache"):
    """
    Create embeddings for filing data during data loading phase.
    Uses disk caching to avoid recreating embeddings and hierarchical data if they already exist.
    
    Args:
        filing_data: Filing data for a specific year
        ticker: Company ticker symbol
        year: Fiscal year
        use_hierarchical: Whether to use hierarchical chunks
        exclude_titles: Whether to exclude title/heading chunks
        cache_dir: Directory for caching embeddings
        
    Returns:
        Dictionary containing processed data with embeddings
    """
    logger.info(f"🧠 Creating embeddings for {ticker} {year}...")
    
    # Create data processor
    processor = DataProcessor()
    
    # Check if embeddings already exist on disk
    if processor.embeddings_exist_on_disk(ticker, year, cache_dir):
        logger.info(f"📥 Found existing embeddings for {ticker} {year}, loading from disk...")
        
        # Load existing embeddings
        if processor.load_embeddings_from_disk(ticker, year, cache_dir):
            logger.info(f"✅ Successfully loaded cached embeddings for {ticker} {year}")
        else:
            logger.warning(f"⚠️  Failed to load cached embeddings, will create new ones")
            # Fall through to create new embeddings
    else:
        logger.info(f"🆕 No cached embeddings found for {ticker} {year}, creating new ones...")
    
    # If we don't have embeddings (either not found or failed to load), create them
    if not processor.has_embeddings():
        logger.info(f"🔧 Creating new embeddings for {ticker} {year}...")
    
    # Add ticker to filing data
    filing_data['ticker'] = ticker
    
    # Prepare chunks
    processor.prepare_chunks(filing_data, use_hierarchical, exclude_titles)
    
    # Create embeddings
    processor.create_embeddings()
    
    # Save embeddings and hierarchical data to disk for future use
    logger.info(f"💾 Saving embeddings and hierarchical data to disk for future use...")
    save_path = processor.save_embeddings_to_disk(ticker, year, cache_dir, filing_data)
    if save_path:
        logger.info(f"✅ Embeddings and hierarchical data saved to {save_path}")
    else:
        logger.warning(f"⚠️  Failed to save embeddings to disk")
    
    # Get processed data
    chunks = processor.get_chunks()
    embeddings = processor.get_embeddings()
    tfidf_vectorizer = processor.get_tfidf_vectorizer()
    tfidf_matrix = processor.get_tfidf_matrix()
    cross_encoder = processor.get_cross_encoder()
    embedding_model = processor.get_embedding_model()
    
    # Create enhanced filing data with embeddings
    enhanced_filing_data = {
        **filing_data,  # Original data
        'ticker': ticker,
        'fiscal_year': year,
        'processed_chunks': chunks,
        'embeddings': embeddings,
        'tfidf_vectorizer': tfidf_vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'cross_encoder': cross_encoder,
        'embedding_model': embedding_model,
        'data_processor': processor,  # Keep reference to processor
        'embedding_metadata': {
            'num_chunks': len(chunks),
            'embedding_shape': embeddings.shape if embeddings is not None else None,
            'tfidf_shape': tfidf_matrix.shape if tfidf_matrix is not None else None,
            'created_at': datetime.now().isoformat(),
            'cached': processor.embeddings_exist_on_disk(ticker, year, cache_dir),
            'hierarchical_cached': processor.hierarchical_data_exist_on_disk(ticker, year, cache_dir)
        }
    }
    
    logger.info(f"✅ Ready with embeddings for {ticker} {year}: {len(chunks)} chunks, {embeddings.shape if embeddings is not None else 'None'} embeddings")
    
    return enhanced_filing_data

# =============================================================================
# FINQUAL INTEGRATION
# =============================================================================

def fetch_finqual_statements(ticker, year):
    """
    Fetch clean financial statements from finqual for a given ticker and year.

    Args:
        ticker: Company ticker symbol
        year: Fiscal year

    Returns:
        Dictionary with finqual data:
        {
            'income_statement': polars DataFrame or None,
            'balance_sheet': polars DataFrame or None,
            'cash_flow': polars DataFrame or None,
            'income_statement_str': string representation or None,
            'balance_sheet_str': string representation or None,
            'cash_flow_str': string representation or None
        }
    """
    try:
        import finqual as fq
        logger.info(f"📊 Fetching finqual statements for {ticker} {year}...")

        finqual_data = {
            'income_statement': None,
            'balance_sheet': None,
            'cash_flow': None,
            'income_statement_str': None,
            'balance_sheet_str': None,
            'cash_flow_str': None
        }

        company = fq.Finqual(ticker)

        # Helper to convert Polars DataFrame to full string (no truncation)
        def df_to_full_string(df):
            """Convert Polars DataFrame to complete string without truncation."""
            if df is None or df.is_empty():
                return None
            
            # Method 1: Use write_csv to get complete data as formatted table
            # This ensures ALL rows are included
            try:
                import io
                # Get column names
                cols = df.columns
                
                # Build a formatted table string with ALL rows
                lines = []
                
                # Header
                lines.append("| " + " | ".join(cols) + " |")
                lines.append("|" + "|".join(["---" for _ in cols]) + "|")
                
                # All data rows (no truncation!)
                for row in df.iter_rows():
                    formatted_row = []
                    for val in row:
                        if val is None:
                            formatted_row.append("N/A")
                        elif isinstance(val, float):
                            # Format large numbers with commas, keep decimals for small numbers
                            if abs(val) >= 1e6:
                                formatted_row.append(f"{val:,.0f}")
                            elif abs(val) >= 1:
                                formatted_row.append(f"{val:,.2f}")
                            else:
                                formatted_row.append(f"{val:.4f}")
                        else:
                            formatted_row.append(str(val))
                    lines.append("| " + " | ".join(formatted_row) + " |")
                
                result = f"Financial Statement ({df.shape[0]} rows x {df.shape[1]} columns)\n"
                result += "\n".join(lines)
                return result
                
            except Exception as e:
                # Fallback: use Polars with max rows config
                import polars as pl
                with pl.Config(tbl_rows=-1, tbl_cols=-1):
                    return str(df)

        # Fetch income statement
        try:
            df = company.income_stmt(year)
            if df is not None and not df.is_empty():
                finqual_data['income_statement'] = df
                finqual_data['income_statement_str'] = df_to_full_string(df)
                logger.info(f"   ✅ Income statement: {df.shape}")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not fetch income statement: {e}")

        # Fetch balance sheet
        try:
            df = company.balance_sheet(year)
            if df is not None and not df.is_empty():
                finqual_data['balance_sheet'] = df
                finqual_data['balance_sheet_str'] = df_to_full_string(df)
                logger.info(f"   ✅ Balance sheet: {df.shape}")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not fetch balance sheet: {e}")

        # Fetch cash flow
        try:
            df = company.cash_flow(year)
            if df is not None and not df.is_empty():
                finqual_data['cash_flow'] = df
                finqual_data['cash_flow_str'] = df_to_full_string(df)
                logger.info(f"   ✅ Cash flow: {df.shape}")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not fetch cash flow: {e}")

        # Count successful fetches
        count = sum(1 for v in [finqual_data['income_statement'],
                                finqual_data['balance_sheet'],
                                finqual_data['cash_flow']] if v is not None)
        logger.info(f"📊 Fetched {count}/3 finqual statements for {ticker} {year}")

        return finqual_data

    except ImportError:
        logger.warning("⚠️  finqual not available - skipping finqual data fetch")
        return None
    except Exception as e:
        logger.error(f"❌ Error fetching finqual data for {ticker} {year}: {e}")
        return None


def save_finqual_to_cache(ticker, year, finqual_data, cache_dir="embeddings_cache"):
    """
    Save finqual statements to cache directory.

    Args:
        ticker: Company ticker symbol
        year: Fiscal year
        finqual_data: Dictionary from fetch_finqual_statements
        cache_dir: Directory for cached data
    """
    if finqual_data is None:
        return

    save_dir = Path(cache_dir) / ticker / str(year)
    save_dir.mkdir(parents=True, exist_ok=True)

    finqual_cache_file = save_dir / "finqual_statements.pkl"

    # Save the string representations (DataFrames are not easily pickle-able)
    cache_data = {
        'income_statement_str': finqual_data.get('income_statement_str'),
        'balance_sheet_str': finqual_data.get('balance_sheet_str'),
        'cash_flow_str': finqual_data.get('cash_flow_str'),
        'ticker': ticker,
        'year': year,
        'cached_at': datetime.now().isoformat()
    }

    with open(finqual_cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    logger.info(f"💾 Saved finqual statements to {finqual_cache_file}")


def load_finqual_from_cache(ticker, year, cache_dir="embeddings_cache"):
    """
    Load finqual statements from cache directory.

    Args:
        ticker: Company ticker symbol
        year: Fiscal year
        cache_dir: Directory for cached data

    Returns:
        Dictionary with string representations of statements, or None if not found
    """
    finqual_cache_file = Path(cache_dir) / ticker / str(year) / "finqual_statements.pkl"

    if not finqual_cache_file.exists():
        return None

    try:
        with open(finqual_cache_file, 'rb') as f:
            finqual_data = pickle.load(f)
        logger.info(f"📥 Loaded finqual statements from cache for {ticker} {year}")
        return finqual_data
    except Exception as e:
        logger.warning(f"⚠️  Could not load finqual cache for {ticker} {year}: {e}")
        return None


def load_filing_data_from_cache(ticker, year, cache_dir="embeddings_cache"):
    """
    Load filing data directly from disk cache without downloading or parsing.
    This is much faster for subsequent runs.

    Args:
        ticker: Company ticker symbol
        year: Fiscal year
        cache_dir: Directory for cached data

    Returns:
        Dictionary with filing data and embeddings, or None if not found
    """
    logger.info(f"📥 Loading cached filing data for {ticker} {year}...")
    
    # Create data processor
    processor = DataProcessor()
    
    # Check if both embeddings and hierarchical data exist
    if not processor.embeddings_exist_on_disk(ticker, year, cache_dir):
        logger.warning(f"⚠️  No cached embeddings found for {ticker} {year}")
        return None
    
    if not processor.hierarchical_data_exist_on_disk(ticker, year, cache_dir):
        logger.warning(f"⚠️  No cached hierarchical data found for {ticker} {year}")
        return None
    
    # Load embeddings
    if not processor.load_embeddings_from_disk(ticker, year, cache_dir):
        logger.error(f"❌ Failed to load embeddings for {ticker} {year}")
        return None
    
    # Load hierarchical data
    hierarchical_data = processor.load_hierarchical_data_from_disk(ticker, year, cache_dir)
    if not hierarchical_data:
        logger.error(f"❌ Failed to load hierarchical data for {ticker} {year}")
        return None
    
    # Get processed data
    chunks = processor.get_chunks()
    embeddings = processor.get_embeddings()
    tfidf_vectorizer = processor.get_tfidf_vectorizer()
    tfidf_matrix = processor.get_tfidf_matrix()
    cross_encoder = processor.get_cross_encoder()
    embedding_model = processor.get_embedding_model()
    tables = processor.get_tables()
    financial_tables_index = processor.get_financial_tables_index()

    # Load finqual statements from cache
    finqual_data = load_finqual_from_cache(ticker, year, cache_dir)

    # Create enhanced filing data
    enhanced_filing_data = {
        **hierarchical_data,  # Include hierarchical data
        'ticker': ticker,
        'fiscal_year': year,
        'processed_chunks': chunks,
        'embeddings': embeddings,
        'tfidf_vectorizer': tfidf_vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'cross_encoder': cross_encoder,
        'embedding_model': embedding_model,
        'tables': tables,
        'financial_tables_index': financial_tables_index,
        'finqual_statements': finqual_data,  # Add finqual data
        'data_processor': processor,
        'embedding_metadata': {
            'num_chunks': len(chunks),
            'embedding_shape': embeddings.shape if embeddings is not None else None,
            'tfidf_shape': tfidf_matrix.shape if tfidf_matrix is not None else None,
            'created_at': datetime.now().isoformat(),
            'cached': True,
            'hierarchical_cached': True,
            'loaded_from_cache': True,
            'finqual_cached': finqual_data is not None
        }
    }
    
    logger.info(f"✅ Successfully loaded cached data for {ticker} {year}")
    logger.info(f"   - {len(chunks)} processed chunks")
    logger.info(f"   - {embeddings.shape if embeddings is not None else 'None'} embeddings")
    logger.info(f"   - {hierarchical_data.get('num_hierarchical_chunks', 0)} hierarchical chunks")
    logger.info(f"   - {hierarchical_data.get('num_contextual_chunks', 0)} contextual chunks")
    
    return enhanced_filing_data

def download_and_extract_10k_with_embeddings(ticker, start_year, end_year, use_hierarchical=True, exclude_titles=True, cache_dir="embeddings_cache"):
    """
    Download and extract 10-K with embeddings for a given ticker and year range.
    This is the main function that combines data loading and embedding creation.
    Uses disk caching to avoid recreating embeddings.
    
    Args:
        ticker: Company ticker symbol
        start_year: Start year for data extraction
        end_year: End year for data extraction
        use_hierarchical: Whether to use hierarchical chunks
        exclude_titles: Whether to exclude title/heading chunks
        cache_dir: Directory for caching embeddings
        
    Returns:
        Dictionary organized by fiscal year with embeddings included
    """
    logger.info(f"📥 Downloading and processing {ticker} from {start_year} to {end_year} with embeddings...")
    
    # First, download and extract raw data
    raw_filing_data = download_and_extract_10k(ticker, start_year, end_year)
    
    # Process each year to create embeddings
    enhanced_filing_data = {}
    
    for year, year_data in raw_filing_data.items():
        logger.info(f"🔧 Processing {ticker} {year} with embeddings...")
        
        try:
            enhanced_year_data = create_embeddings_for_filing_data(
                year_data, 
                ticker, 
                year, 
                use_hierarchical, 
                exclude_titles,
                cache_dir
            )
            enhanced_filing_data[year] = enhanced_year_data
            
        except Exception as e:
            logger.error(f"❌ Failed to create embeddings for {ticker} {year}: {e}")
            # Still include raw data even if embeddings fail
            enhanced_filing_data[year] = {
                **year_data,
                'ticker': ticker,
                'fiscal_year': year,
                'embedding_error': str(e)
            }
    
    logger.info(f"✅ Completed processing {ticker} with embeddings for {len(enhanced_filing_data)} years")
    return enhanced_filing_data

def download_and_extract_fiscal_years(ticker, fiscal_years, use_hierarchical=True, exclude_titles=True, cache_dir="embeddings_cache"):
    """
    Download and extract 10-K filings for multiple fiscal years.
    Automatically calculates filing year range to account for fiscal year vs filing year mismatches.
    
    Args:
        ticker: Company ticker symbol
        fiscal_years: List of fiscal years to load
        use_hierarchical: Whether to use hierarchical chunks
        exclude_titles: Whether to exclude title/heading chunks
        cache_dir: Directory for caching embeddings
        
    Returns:
        Dictionary organized by fiscal year with embeddings included
    """
    if not fiscal_years:
        logger.error("❌ No fiscal years specified")
        return {}
    
    # Calculate filing year range with buffer
    start_year, end_year = calculate_filing_year_range(fiscal_years)
    logger.info(f"📅 Requested fiscal years: {fiscal_years}")
    logger.info(f"📅 Searching filing years: {start_year} to {end_year} (with buffer for fiscal year mismatches)")
    
    # Download and extract raw data
    raw_filing_data = download_and_extract_10k(ticker, start_year, end_year)
    
    # Filter to only include requested fiscal years
    filtered_filing_data = {}
    for fiscal_year in fiscal_years:
        if fiscal_year in raw_filing_data:
            filtered_filing_data[fiscal_year] = raw_filing_data[fiscal_year]
            logger.info(f"✅ Found fiscal year {fiscal_year} in filings")
        else:
            logger.warning(f"⚠️  Fiscal year {fiscal_year} not found in downloaded filings")
    
    # If no requested fiscal years were found, show what was actually found
    if not filtered_filing_data:
        logger.warning(f"⚠️  No requested fiscal years found. Available fiscal years: {list(raw_filing_data.keys())}")
        logger.warning(f"⚠️  Requested fiscal years: {fiscal_years}")
        logger.warning(f"⚠️  This might be due to fiscal year vs filing year mismatches")
    
    # Process each year to create embeddings
    enhanced_filing_data = {}
    
    for fiscal_year, year_data in filtered_filing_data.items():
        logger.info(f"🔧 Processing {ticker} {fiscal_year} with embeddings...")
        
        try:
            enhanced_year_data = create_embeddings_for_filing_data(
                year_data, 
                ticker, 
                fiscal_year, 
                use_hierarchical, 
                exclude_titles,
                cache_dir
            )
            enhanced_filing_data[fiscal_year] = enhanced_year_data
            
        except Exception as e:
            logger.error(f"❌ Failed to create embeddings for {ticker} {fiscal_year}: {e}")
            # Still include raw data even if embeddings fail
            enhanced_filing_data[fiscal_year] = {
                **year_data,
                'ticker': ticker,
                'fiscal_year': fiscal_year,
                'embedding_error': str(e)
            }
    
    logger.info(f"✅ Completed processing {ticker} with embeddings for {len(enhanced_filing_data)} fiscal years")
    return enhanced_filing_data


def find_fiscal_year_for_ticker(ticker, filing_year, cache_dir="embeddings_cache"):
    """
    Find the fiscal year associated with a ticker and filing year.
    This is useful when you have a filing year but need to know the fiscal year for caching.
    
    Args:
        ticker: Company ticker symbol
        filing_year: Filing year to search for
        cache_dir: Directory for cached data
        
    Returns:
        Fiscal year string if found, None otherwise
    """
    logger.info(f"🔍 Finding fiscal year for {ticker} filing year {filing_year}...")
    
    try:
        # Download and extract to find fiscal year
        filing_data = download_and_extract_10k(ticker, filing_year, filing_year)
        
        if filing_data:
            fiscal_years = list(filing_data.keys())
            if fiscal_years:
                fiscal_year = fiscal_years[0]
                logger.info(f"✅ Found fiscal year {fiscal_year} for {ticker} filing year {filing_year}")
                return fiscal_year
        
        logger.warning(f"⚠️  No fiscal year found for {ticker} filing year {filing_year}")
        return None
        
    except Exception as e:
        logger.error(f"❌ Failed to find fiscal year for {ticker} filing year {filing_year}: {e}")
        return None


def collect_company_stats(ticker, years, cache_dir="embeddings_cache"):
    """
    Collect statistics for a company's cached data across multiple years

    Args:
        ticker: Company ticker symbol
        years: List of fiscal years
        cache_dir: Directory for cached data

    Returns:
        Dictionary with statistics for the company
    """
    processor = DataProcessor()
    stats = {
        'ticker': ticker,
        'years_processed': [],
        'years_with_zero_tables': [],  # Track years with zero table chunks
        'total_text_chunks': 0,
        'total_table_chunks': 0,
        'total_tables': 0,
        'years_stats': {}
    }

    for year in years:
        if processor.embeddings_exist_on_disk(ticker, year, cache_dir):
            try:
                # Load metadata to get stats
                load_dir = Path(cache_dir) / ticker / str(year)
                metadata_file = load_dir / "metadata.pkl"

                if metadata_file.exists():
                    with open(metadata_file, "rb") as f:
                        metadata = pickle.load(f)

                    chunk_types = metadata.get('chunk_types', {})
                    text_chunks = chunk_types.get('text', 0)
                    table_chunks = chunk_types.get('table', 0)

                    # Load tables to count them
                    tables_file = load_dir / "tables.pkl"
                    num_tables = 0
                    if tables_file.exists():
                        with open(tables_file, "rb") as f:
                            tables = pickle.load(f)
                            num_tables = len(tables)

                    stats['years_processed'].append(year)
                    stats['total_text_chunks'] += text_chunks
                    stats['total_table_chunks'] += table_chunks
                    stats['total_tables'] += num_tables
                    stats['years_stats'][year] = {
                        'text_chunks': text_chunks,
                        'table_chunks': table_chunks,
                        'tables': num_tables
                    }

                    # FLAG: Track years with zero table chunks
                    if table_chunks == 0:
                        stats['years_with_zero_tables'].append(year)
                        logger.warning(f"⚠️  {ticker} {year}: ZERO TABLE CHUNKS FOUND")
            except Exception as e:
                logger.warning(f"⚠️  Error collecting stats for {ticker} {year}: {e}")

    return stats

def print_loading_statistics(company_stats):
    """
    Print comprehensive statistics for all loaded companies

    Args:
        company_stats: Dictionary mapping company names to their statistics
    """
    if not company_stats:
        logger.info("📊 No statistics to display")
        return

    logger.info("\n" + "="*100)
    logger.info("📊 FINAL DATA LOADING STATISTICS")
    logger.info("="*100)

    total_companies = len(company_stats)
    total_text_chunks = 0
    total_table_chunks = 0
    total_tables = 0
    companies_with_zero_tables = []

    for company_name, stats in company_stats.items():
        ticker = stats['ticker']
        years_processed = stats['years_processed']
        years_with_zero_tables = stats.get('years_with_zero_tables', [])

        logger.info(f"\n📈 {company_name} ({ticker})")
        logger.info(f"   Years processed: {len(years_processed)} - {years_processed}")
        logger.info(f"   Total text chunks: {stats['total_text_chunks']}")
        logger.info(f"   Total table chunks: {stats['total_table_chunks']}")
        logger.info(f"   Total tables: {stats['total_tables']}")

        # FLAG: Highlight years with zero table chunks
        if years_with_zero_tables:
            logger.warning(f"   ⚠️  YEARS WITH ZERO TABLE CHUNKS: {years_with_zero_tables}")
            for year in years_with_zero_tables:
                companies_with_zero_tables.append((company_name, ticker, year))

        # Show breakdown by year
        if stats['years_stats']:
            logger.info(f"   Year breakdown:")
            for year, year_stats in sorted(stats['years_stats'].items()):
                zero_flag = " ⚠️  ZERO TABLES!" if year_stats['table_chunks'] == 0 else ""
                logger.info(f"      {year}: {year_stats['text_chunks']} text chunks, {year_stats['table_chunks']} table chunks, {year_stats['tables']} tables{zero_flag}")

        total_text_chunks += stats['total_text_chunks']
        total_table_chunks += stats['total_table_chunks']
        total_tables += stats['total_tables']

    logger.info("\n" + "-"*100)
    logger.info("📊 AGGREGATE STATISTICS")
    logger.info("-"*100)
    logger.info(f"   Total companies: {total_companies}")
    logger.info(f"   Total text chunks: {total_text_chunks}")
    logger.info(f"   Total table chunks: {total_table_chunks}")
    logger.info(f"   Total tables: {total_tables}")

    # FLAG: Summary of companies/years with zero table chunks
    if companies_with_zero_tables:
        logger.warning("\n" + "!"*100)
        logger.warning("⚠️  COMPANIES/YEARS WITH ZERO TABLE CHUNKS:")
        logger.warning("!"*100)
        for company_name, ticker, year in companies_with_zero_tables:
            logger.warning(f"   - {company_name} ({ticker}): {year}")
        logger.warning("!"*100)

    logger.info("="*100 + "\n")

def load_financebench_company_data(company_name, ticker, years, cache_dir="embeddings_cache"):
    """Load data for a specific FinanceBench company"""
    logger.info(f"📥 Loading {company_name} ({ticker}) for years {years}...")

    try:
        result = download_and_extract_fiscal_years(
            ticker=ticker,
            fiscal_years=years,
            use_hierarchical=True,
            exclude_titles=True,
            cache_dir=cache_dir
        )

        # Log results
        success_count = 0
        for fiscal_year, year_data in result.items():
            if year_data.get('embedding_error') is None:
                logger.info(f"✅ Successfully processed {company_name} ({ticker}) fiscal year {fiscal_year}")
                success_count += 1
            else:
                logger.error(f"❌ Failed to create embeddings for {company_name} ({ticker}) fiscal year {fiscal_year}: {year_data['embedding_error']}")

        logger.info(f"🎯 {company_name} ({ticker}) completed: {success_count}/{len(years)} years successful")
        return success_count > 0

    except Exception as e:
        logger.error(f"❌ Error loading {company_name} ({ticker}): {e}")
        return False

def load_all_financebench_companies(years, cache_dir="embeddings_cache", use_specific_years=False, specific_company_years=None):
    """
    Load data for all FinanceBench companies

    Args:
        years: Default years to load for all companies (used when use_specific_years is False)
        cache_dir: Directory for caching embeddings
        use_specific_years: If True, use specific_company_years dictionary for company-year combinations
        specific_company_years: Dictionary mapping company names to lists of years {company_name: [years]}
    """
    if use_specific_years and specific_company_years:
        logger.info(f"🎯 Loading data for specific company-year combinations...")
        logger.info(f"   Companies to load: {list(specific_company_years.keys())}")
    else:
        logger.info(f"🎯 Loading data for all FinanceBench companies for years {years}...")

    companies_to_load = FINANCEBENCH_COMPANIES
    if FINANCEBENCH_COMPANY_FILTER:
        # Handle both single company string and list of companies
        if isinstance(FINANCEBENCH_COMPANY_FILTER, str):
            company_filter = [FINANCEBENCH_COMPANY_FILTER]
        else:
            company_filter = FINANCEBENCH_COMPANY_FILTER

        # Filter companies to only include those in the filter list
        companies_to_load = {name: ticker for name, ticker in FINANCEBENCH_COMPANIES.items() if name in company_filter}

        if not companies_to_load:
            logger.error(f"❌ No companies found matching filter: {FINANCEBENCH_COMPANY_FILTER}")
            return {}

        logger.info(f"🔍 Filtering to specific companies: {list(companies_to_load.keys())}")

    # If using specific company-years, filter to only those companies
    if use_specific_years and specific_company_years:
        companies_to_load = {name: ticker for name, ticker in companies_to_load.items() if name in specific_company_years}
        if not companies_to_load:
            logger.error(f"❌ No companies found in specific_company_years")
            return {}

    total_companies = len(companies_to_load)
    successful_companies = 0
    company_stats = {}

    for i, (company_name, ticker) in enumerate(companies_to_load.items(), 1):
        logger.info(f"📊 Processing company {i}/{total_companies}: {company_name} ({ticker})")

        # Determine which years to load for this company
        if use_specific_years and specific_company_years and company_name in specific_company_years:
            company_years = specific_company_years[company_name]
            logger.info(f"   Loading specific years: {company_years}")
        else:
            company_years = years

        if load_financebench_company_data(company_name, ticker, company_years, cache_dir):
            successful_companies += 1
            # Collect stats for this company
            company_stats[company_name] = collect_company_stats(ticker, company_years, cache_dir)

        logger.info("-" * 80)

    logger.info(f"🎯 FinanceBench loading completed: {successful_companies}/{total_companies} companies successful")

    # Print statistics at the end
    print_loading_statistics(company_stats)

    return company_stats

def main():
    """Main function using configuration variables."""
    logger.info("🚀 Starting SEC Filings Data Loader")
    logger.info(f"📊 Configuration:")
    logger.info(f"   - Ticker: {TICKER}")
    logger.info(f"   - Fiscal Years: {FISCAL_YEARS}")
    logger.info(f"   - Load FinanceBench: {LOAD_FINANCEBENCH_COMPANIES}")
    logger.info(f"   - FinanceBench Company Filter: {FINANCEBENCH_COMPANY_FILTER}")
    logger.info(f"   - FinanceBench Years: {FINANCEBENCH_YEARS}")
    logger.info(f"   - Use Specific Company Years: {USE_SPECIFIC_COMPANY_YEARS}")
    if USE_SPECIFIC_COMPANY_YEARS:
        logger.info(f"   - Specific Company Years: {SPECIFIC_COMPANY_YEARS}")
    logger.info(f"   - Cache Dir: {CACHE_DIR}")
    logger.info(f"   - Check Cache Only: {CHECK_CACHE_ONLY}")
    
    # Check cache status if requested
    if CHECK_CACHE_ONLY:
        logger.info("🔍 Checking cache status...")
        processor = DataProcessor()
        cache_path = Path(CACHE_DIR)
        
        if not cache_path.exists():
            logger.info(f"📁 Cache directory {CACHE_DIR} does not exist")
            return
        
        if LOAD_FINANCEBENCH_COMPANIES:
            # Check cache for FinanceBench companies
            companies_to_check = FINANCEBENCH_COMPANIES
            if FINANCEBENCH_COMPANY_FILTER:
                companies_to_check = {FINANCEBENCH_COMPANY_FILTER: FINANCEBENCH_COMPANIES[FINANCEBENCH_COMPANY_FILTER]}
            
            for company_name, ticker in companies_to_check.items():
                logger.info(f"📊 {company_name} ({ticker}):")
                for year in FINANCEBENCH_YEARS:
                    embeddings_exist = processor.embeddings_exist_on_disk(ticker, year, CACHE_DIR)
                    hierarchical_exist = processor.hierarchical_data_exist_on_disk(ticker, year, CACHE_DIR)
                    logger.info(f"   {year}: embeddings={embeddings_exist}, hierarchical={hierarchical_exist}")
        else:
            # Check cache for single ticker
            for fiscal_year in FISCAL_YEARS:
                embeddings_exist = processor.embeddings_exist_on_disk(TICKER, fiscal_year, CACHE_DIR)
                hierarchical_exist = processor.hierarchical_data_exist_on_disk(TICKER, fiscal_year, CACHE_DIR)
                logger.info(f"📊 {TICKER} {fiscal_year}: embeddings={embeddings_exist}, hierarchical={hierarchical_exist}")
        return
    
    if LOAD_FINANCEBENCH_COMPANIES:
        # Load data for FinanceBench companies
        load_all_financebench_companies(
            FINANCEBENCH_YEARS,
            CACHE_DIR,
            use_specific_years=USE_SPECIFIC_COMPANY_YEARS,
            specific_company_years=SPECIFIC_COMPANY_YEARS if USE_SPECIFIC_COMPANY_YEARS else None
        )
    else:
        # Load data for specified fiscal years
        logger.info(f"📥 Loading {TICKER} for fiscal years {FISCAL_YEARS}...")
        filing_data = download_and_extract_fiscal_years(
            ticker=TICKER,
            fiscal_years=FISCAL_YEARS,
            use_hierarchical=True,
            exclude_titles=True,
            cache_dir=CACHE_DIR
        )
    
    if not LOAD_FINANCEBENCH_COMPANIES:
        # Log results for single ticker
        successful_years = 0
        total_years = len(filing_data)
        for fiscal_year, year_data in filing_data.items():
            if 'embedding_error' not in year_data:
                successful_years += 1
                metadata = year_data.get('embedding_metadata', {})
                logger.info(f"✅ Successfully processed {TICKER} fiscal year {fiscal_year}")
                logger.info(f"   - {metadata.get('num_chunks', 0)} processed chunks")
                logger.info(f"   - {metadata.get('embedding_shape', 'None')} embeddings")
            else:
                logger.error(f"❌ Failed to create embeddings for {TICKER} fiscal year {fiscal_year}: {year_data['embedding_error']}")
        
        logger.info(f"📊 Final summary: {successful_years}/{total_years} fiscal years successful")
        
        # Print fiscal years found
        print("\n\n")
        print("\n\n")
        print("FISCAL YEARS SELECTED: ", list(filing_data.keys()))
        print("\n\n")
        print("\n\n")
    
    # Final summary
    logger.info("🎯 Data loading completed!")

if __name__ == "__main__":
    main()