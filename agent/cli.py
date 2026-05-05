#!/usr/bin/env python3
"""
CLI debug runner for FilesystemResearchAgent.

Usage:
    python -m agent.cli "what was NVDA's data center revenue trend?"
    python -m agent.cli -v "..."          verbose: full event payloads
    python -m agent.cli --raw "..."       raw event JSON, one per line
    echo "..." | python -m agent.cli

stderr: progress / tool calls / reasoning
stdout: final answer
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Any, Dict


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


def _fmt_args(args: Dict[str, Any]) -> str:
    """Compact one-line repr of tool args."""
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 80:
            v = v[:77] + "..."
        parts.append(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}")
    return ", ".join(parts)


def _fmt_event(event: Dict[str, Any], verbose: bool) -> str:
    et = event.get("type", "?")

    if et == "progress":
        return f"[progress] {event.get('message', '')}"

    if et == "llm_call":
        phase = event.get("phase", "")
        n = event.get("call_num")
        msgs = event.get("messages")
        used = event.get("tool_calls_so_far")
        suffix = f" [{phase}]" if phase else ""
        if msgs is not None:
            return f"[llm #{n}{suffix}] msgs={msgs}, tools_used={used}"
        return f"[llm #{n}{suffix}]"

    if et == "tool_start":
        return f"[tool→ #{event.get('call_num')}] {event.get('name')}({_fmt_args(event.get('args', {}))})"

    if et == "tool_end":
        chars = event.get("result_chars", 0)
        prev = event.get("result_preview", "")
        if verbose:
            return f"[tool← #{event.get('call_num')}] {event.get('name')}  ({chars} chars)\n        {prev}"
        return f"[tool← #{event.get('call_num')}] {event.get('name')}  ({chars} chars)"

    if et == "error":
        return f"[ERROR] {event.get('message')}"

    if et == "token":
        return ""  # handled separately

    if et == "result":
        return ""  # handled separately

    if verbose:
        return f"[{et}] {json.dumps(event, default=str)[:200]}"
    return f"[{et}]"


async def _run(question: str, args) -> int:
    from .agent import FilesystemResearchAgent

    print(f"\n→ Question: {question}\n", file=sys.stderr)
    agent = FilesystemResearchAgent(
        model=args.model,
        max_tool_calls=args.budget,
    )

    in_token_stream = False
    final_event: Dict[str, Any] = {}
    error_seen = False
    t0 = time.time()

    async for event in agent.run(question):
        if args.raw:
            print(json.dumps(event, default=str))
            continue

        et = event.get("type")

        if et == "result":
            final_event = event
            continue
        if et == "error":
            error_seen = True
            print(f"\n\n❌ {event.get('message')}", file=sys.stderr)
            continue

        if et == "token":
            if not in_token_stream:
                print("\n\n=== ANSWER ===\n", flush=True)
                in_token_stream = True
            sys.stdout.write(event.get("content", ""))
            sys.stdout.flush()
            continue

        if in_token_stream:
            print("\n", flush=True)
            in_token_stream = False

        line = _fmt_event(event, args.verbose)
        if line:
            print(line, file=sys.stderr, flush=True)

    elapsed = time.time() - t0

    if error_seen:
        return 1

    if final_event and not args.raw:
        if not in_token_stream:
            print("\n=== ANSWER ===\n")
            print(final_event.get("answer", ""))
        tcs = final_event.get("tool_calls", 0)
        llms = final_event.get("llm_calls", 0)
        print(
            f"\n\n--- {elapsed:.1f}s | {tcs} tool call(s) | {llms} LLM call(s) ---",
            file=sys.stderr,
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filesystem-only financial research agent (gpt-5.4-mini).",
    )
    parser.add_argument("question", nargs="*", help="Question; if omitted reads from stdin.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose tool output.")
    parser.add_argument("--raw", action="store_true", help="Emit raw event JSON.")
    parser.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    parser.add_argument("--budget", type=int, default=25, help="Max tool calls (default 25).")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if args.question:
        q = " ".join(args.question).strip()
    else:
        q = sys.stdin.read().strip()
    if not q:
        parser.error("no question provided (pass as args or via stdin)")

    return asyncio.run(_run(q, args))


if __name__ == "__main__":
    sys.exit(main())
