"""
DuckDB Module for Financial Data Analysis

This module contains the FinancialDataAnalyzer class and related utilities
for analyzing financial data using DuckDB.
"""

from .main_duckdb import (
    FinancialDataAnalyzer,
    SystemInitializationError,
    QueryProcessingError,
    SchemaValidationError
)
from .qualitative_screener import QualitativeScreener

__all__ = [
    'FinancialDataAnalyzer',
    'SystemInitializationError',
    'QueryProcessingError',
    'SchemaValidationError',
    'QualitativeScreener',
]

__version__ = '1.0.0'
