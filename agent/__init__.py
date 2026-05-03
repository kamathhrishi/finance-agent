#!/usr/bin/env python3
"""
agent/ — public entry point for the StrataLens AI agent.

There used to be three implementations selectable via env vars (legacy
chunk-RAG `RAGAgent`, multi-agent ReAct `OrchestratorAgent`, and the
filesystem-research `FilesystemResearchOrchestrator`). The first two have
been retired — the platform now ships only the FS research agent.

For the historical context behind the decision, see the original blog
post on the multi-agent / chunk-RAG approach (linked from `agent/README.md`)
and the design notes in `fs_research_agent/README.md`.

Public API:
  - `Agent` / `AgentSystem` — alias for `FilesystemResearchOrchestrator`
  - `create_agent()` — factory that constructs a fresh agent instance

Sub-packages kept under `agent/`:
  - `screener/` — DuckDB-backed stock screener (independent of the agent;
                  used by the screener router and qualitative-screen flow)
"""

import os as _os

# Hard guard against any leftover deploys still setting USE_FS_RESEARCH_AGENT=false.
# The legacy agents are gone; falling back to them is no longer possible.
_explicit = _os.getenv("USE_FS_RESEARCH_AGENT", "").strip().lower()
if _explicit in ("0", "false", "no"):
    raise RuntimeError(
        "USE_FS_RESEARCH_AGENT=false is no longer supported — the legacy "
        "agent (RAGAgent / OrchestratorAgent) has been removed. The "
        "filesystem-research agent is now the only implementation. "
        "Unset USE_FS_RESEARCH_AGENT (or set it to true) and redeploy."
    )

from fs_research_agent.orchestrator_adapter import FilesystemResearchOrchestrator

# Public API
Agent = FilesystemResearchOrchestrator
AgentSystem = FilesystemResearchOrchestrator


def create_agent():
    """Construct a fresh agent instance."""
    return FilesystemResearchOrchestrator()


__all__ = ["Agent", "AgentSystem", "create_agent"]
