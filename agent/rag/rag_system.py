#!/usr/bin/env python3
"""
RAG System - Backward Compatibility Wrapper

This module provides backward compatibility for code that imports from rag_system.
The architecture has been reorganized:
- rag/: Core RAG functionality (search, retrieval, embeddings)
- agent/: Agentic orchestration (iterative improvement, evaluation)

This file re-exports AgentSystem as RAGSystem for backward compatibility.
"""

from agent import Agent as AgentSystem, create_agent as create_agent_system

# Backward compatibility: RAGSystem now points to AgentSystem
RAGSystem = AgentSystem
create_rag_system = create_agent_system

# Also export for flexibility
__all__ = ['RAGSystem', 'create_rag_system', 'AgentSystem', 'create_agent_system']
