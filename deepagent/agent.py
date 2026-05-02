"""
NVDA Deep Research Agent using LangChain DeepAgents.

Documents in ./documents/:
  - {year}_10-K.md         — annual 10-K filings (2018–2024)
  - {year}_EX-*.md         — exhibits (insider trading policy, comp plans, etc.)
  - transcript_{year}_Q{q}.txt — earnings call transcripts (2020–2025)

Usage:
    pip install deepagents langchain-openai
    python agent.py
    python agent.py --query "How did NVDA's data center revenue grow from 2022 to 2024?"
"""

import argparse
import os
import time
import random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_openai import ChatOpenAI
from openai import RateLimitError


class RetryingChatOpenAI(ChatOpenAI):
    """ChatOpenAI with automatic exponential-backoff retry on 429s."""

    def _generate(self, *args, **kwargs):
        for attempt in range(8):
            try:
                return super()._generate(*args, **kwargs)
            except RateLimitError as e:
                if attempt == 7:
                    raise
                wait = min(2 ** attempt + random.random(), 60)
                print(f"\n⏳ Rate limited — waiting {wait:.1f}s before retry {attempt + 1}/7...")
                time.sleep(wait)

DOCS_DIR = Path(__file__).parent / "documents"

SYSTEM_PROMPT = """You are a senior financial analyst specializing in NVIDIA (NVDA).

You have access to NVDA's complete document library:
- Annual 10-K filings from 2018 to 2024 (files named {year}_10-K.md)
- SEC exhibits: compensation plans, insider trading policies, subsidiary lists, etc. (files named {year}_EX-*.md)
- Earnings call transcripts from Q1 2020 to Q4 2025 (files named transcript_{year}_Q{q}.txt)

## Research workflow

For any multi-step or multi-year analysis, follow this pattern:

1. **Plan first** — use write_todos to break the task into discrete steps
2. **Check for cached intermediates** — before reading a source doc, check if a scratch file already exists (e.g., `_cache/segments_2022.md`). If it does, read that instead of re-processing the source.
3. **Save as you go** — after extracting data from each document, immediately write findings to a scratch file under `_cache/` (e.g., `write_file("_cache/segments_2022.md", extracted_data)`). Use clear filenames like `_cache/{topic}_{year}.md`.
4. **Read specific documents** — never read everything at once; target the file(s) needed for each step
5. **Compile from scratch files** — once all per-year/per-topic extractions are done, read all `_cache/` files and compile into a final answer
6. **Cite sources** — every number must reference the source file and year it came from

This approach lets you handle large analyses without hitting context limits — intermediate results stay on disk, not in memory.

Be precise, cite sources, and focus on what the documents actually say.
"""


def build_agent():
    llm = RetryingChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17"),
        reasoning_effort="high",
        use_responses_api=True,
    )
    return create_deep_agent(
        model=llm,
        backend=FilesystemBackend(root_dir=str(DOCS_DIR), virtual_mode=False),
        system_prompt=SYSTEM_PROMPT,
    )


def run(query: str):
    agent = build_agent()
    print(f"\nQuery: {query}\n{'='*60}")

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            if not update or not isinstance(update, dict):
                continue
            raw = update.get("messages", []) or []
            try:
                messages = list(raw) if hasattr(raw, "__iter__") and not isinstance(raw, str) else [raw]
            except Exception:
                continue
            for msg in messages:
                # Tool calls (what the agent wants to do)
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        name = tc.get("name", "?")
                        args = tc.get("args", {})
                        # Show first arg value as a preview
                        preview = next(iter(args.values()), "") if args else ""
                        if isinstance(preview, str) and len(preview) > 80:
                            preview = preview[:80] + "..."
                        print(f"\n🔧 [{name}] {preview}")

                # Tool results
                if getattr(msg, "type", None) == "tool":
                    content = msg.content or ""
                    if isinstance(content, str):
                        snippet = content[:200].replace("\n", " ")
                    else:
                        snippet = str(content)[:200]
                    print(f"   → {snippet}")

                # Final assistant text
                if getattr(msg, "type", None) == "ai":
                    content = msg.content
                    if isinstance(content, str) and content.strip():
                        print(f"\n💬 {content}")
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                print(f"\n💬 {block['text']}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="Summarize NVDA's key business segments and revenue trends from 2022 to 2024 based on the 10-K filings.",
    )
    args = parser.parse_args()
    run(args.query)
