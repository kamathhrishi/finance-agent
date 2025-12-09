#!/usr/bin/env python3
"""
Agent - Unified agent for financial Q&A

This module provides a unified agent that handles:
- RAG-based question answering (earnings transcripts, 10-K filings, news)
- Stock screening queries (fundamental data filtering)
- Iterative self-improvement with quality evaluation

Simplified architecture with no circular dependencies.
"""

# New simplified Agent
from .agent import Agent, create_agent

# Backward compatibility: alias AgentSystem to Agent
AgentSystem = Agent
create_agent_system = create_agent

# Keep prompts
from . import prompts

__all__ = ['Agent', 'create_agent', 'AgentSystem', 'create_agent_system', 'prompts']

