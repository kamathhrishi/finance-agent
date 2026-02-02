"""
Centralized Analyzer Manager

Provides a single source of truth for the Financial Analyzer instance
across all routers. Eliminates duplicate getter/setter patterns.
"""

from typing import Optional, Any
from fastapi import HTTPException


class AnalyzerManager:
    """
    Centralized manager for the Financial Analyzer instance.

    Usage:
        # In lifespan.py or main server initialization:
        from app.utils.analyzer_manager import analyzer_manager
        analyzer_manager.set_analyzer(analyzer_instance, available=True)

        # In routers:
        from app.utils.analyzer_manager import get_analyzer
        analyzer = get_analyzer()  # Raises HTTPException if not available
    """

    _instance: Optional[Any] = None
    _available: bool = False

    @classmethod
    def set_analyzer(cls, instance: Any, available: bool = True):
        """
        Set the analyzer instance.

        Args:
            instance: The FinancialDataAnalyzer instance
            available: Whether the analyzer is available for use
        """
        cls._instance = instance
        cls._available = available

    @classmethod
    def get_analyzer(cls) -> Any:
        """
        Get the analyzer instance.

        Returns:
            The FinancialDataAnalyzer instance

        Raises:
            HTTPException: If analyzer is not available or not initialized
        """
        if not cls._available:
            raise HTTPException(
                status_code=503,
                detail="Query features disabled - analyzer not available"
            )
        if cls._instance is None:
            raise HTTPException(
                status_code=503,
                detail="Analyzer not initialized"
            )
        return cls._instance

    @classmethod
    def is_available(cls) -> bool:
        """Check if the analyzer is available."""
        return cls._available and cls._instance is not None

    @classmethod
    def clear(cls):
        """Clear the analyzer instance (for testing or shutdown)."""
        cls._instance = None
        cls._available = False


# Global instance for convenience
analyzer_manager = AnalyzerManager()


# Convenience functions for direct import
def set_analyzer(instance: Any, available: bool = True):
    """Set the analyzer instance."""
    AnalyzerManager.set_analyzer(instance, available)


def get_analyzer() -> Any:
    """
    Get the analyzer instance.

    Returns:
        The FinancialDataAnalyzer instance

    Raises:
        HTTPException: If analyzer is not available
    """
    return AnalyzerManager.get_analyzer()


def is_analyzer_available() -> bool:
    """Check if the analyzer is available."""
    return AnalyzerManager.is_available()
