"""
agent — StrataLens AI's filesystem-only financial research agent.

A ReAct agent (gpt-5.4-mini) that investigates SEC filings using only
ls / read_file / grep / glob over a self-describing folder of markdown,
plus an optional news_search tool. See `agent/README.md` for design notes.

Public API:
  - `Agent` / `AgentSystem` — alias for `FilesystemResearchOrchestrator`
  - `create_agent()` — factory that constructs a fresh agent instance

Sub-packages kept under `agent/`:
  - `screener/` — DuckDB-backed stock screener (independent of the agent;
                  used by the screener router and qualitative-screen flow)
"""

from agent.orchestrator_adapter import FilesystemResearchOrchestrator

# Public API
Agent = FilesystemResearchOrchestrator
AgentSystem = FilesystemResearchOrchestrator


def create_agent():
    """Construct a fresh agent instance."""
    return FilesystemResearchOrchestrator()


__all__ = ["Agent", "AgentSystem", "create_agent"]
