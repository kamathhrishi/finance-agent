#!/usr/bin/env python3
"""
Agent - Unified agent for financial Q&A

This module provides a unified agent that handles:
- RAG-based question answering (earnings transcripts, 10-K filings, news)
- Stock screening queries (fundamental data filtering)
- Iterative self-improvement with quality evaluation

Simplified architecture with no circular dependencies.
"""

import os

from .rag.rag_agent import RAGAgent

# Agent selection (precedence: USE_FS_RESEARCH_AGENT > USE_DEEP_AGENT > legacy RAG)
#   USE_FS_RESEARCH_AGENT=true   → FilesystemResearchOrchestrator (fs_research_agent) — DEFAULT
#   USE_DEEP_AGENT=true          → OrchestratorAgent (deep agent)
#   otherwise                    → RAGAgent (legacy)
#
# Default flipped to TRUE — the FS agent is the production path. Opt-out by
# explicitly setting USE_FS_RESEARCH_AGENT=false to fall back to the legacy
# orchestrator.
_USE_FS_RESEARCH_AGENT = os.getenv("USE_FS_RESEARCH_AGENT", "true").lower() not in ("0", "false", "no", "")
_USE_DEEP_AGENT = os.getenv("USE_DEEP_AGENT", "true").lower() not in ("0", "false", "no")

from .orchestrator import OrchestratorAgent
# Keep DeepRAGAgent importable for backward compat
from .rag.deep_rag_agent import DeepRAGAgent

if _USE_FS_RESEARCH_AGENT:
    from fs_research_agent.orchestrator_adapter import FilesystemResearchOrchestrator
    _ActiveAgent = FilesystemResearchOrchestrator
elif _USE_DEEP_AGENT:
    _ActiveAgent = OrchestratorAgent
else:
    _ActiveAgent = RAGAgent

# Public API: Agent / AgentSystem are whichever implementation is active
Agent = _ActiveAgent
AgentSystem = _ActiveAgent

def create_agent():
    return _ActiveAgent()

# Keep prompts
from . import prompts

__all__ = ['Agent', 'RAGAgent', 'AgentSystem', 'create_agent', 'prompts', 'DeepRAGAgent', 'OrchestratorAgent']
