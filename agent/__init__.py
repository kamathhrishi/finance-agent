#!/usr/bin/env python3
"""
Agent - Simplified unified agent for financial Q&A

This module provides a unified agent that handles:
- RAG-based question answering (earnings transcripts)
- Stock screening queries (fundamental data filtering)
- Iterative improvement (agentic mode)
- Single-pass answers (chat mode)

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

