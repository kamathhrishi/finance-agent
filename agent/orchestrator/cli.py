#!/usr/bin/env python3
"""
CLI debug runner for OrchestratorAgent.

Usage:
    python -m agent.orchestrator.cli "compare MSFT and GOOGL cloud growth"
    python -m agent.orchestrator.cli -v "..."          # verbose: dump all events
    python -m agent.orchestrator.cli --raw "..."       # print raw event JSON, no formatting
    echo "your question" | python -m agent.orchestrator.cli
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Any, Dict


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy third-party loggers unless very verbose
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


def _fmt_event(event: Dict[str, Any], verbose: bool) -> str:
    etype = event.get("type", "?")
    step = event.get("step", "")
    msg = event.get("message", "")
    data = event.get("data", {}) or {}

    if etype == "token":
        return event.get("content", "")

    head = f"[{etype}"
    if step:
        head += f"/{step}"
    head += "]"

    if etype == "reasoning":
        plan = data.get("plan")
        if plan:
            return f"{head} {plan}"
        return f"{head} {msg}"

    if etype == "progress":
        return f"{head} {msg}"

    if etype == "tool_start":
        name = data.get("tool") or data.get("name") or ""
        args = data.get("args") or data.get("arguments") or {}
        return f"{head} {name}({json.dumps(args, default=str)[:240]})"

    if etype == "tool_end":
        name = data.get("tool") or data.get("name") or ""
        result_preview = str(data.get("result_preview") or data.get("preview") or "")[:200]
        return f"{head} {name} → {result_preview}"

    if etype == "error":
        return f"{head} {msg or data}"

    if etype == "result":
        # Handled separately at end
        return ""

    if verbose:
        return f"{head} {msg} | {json.dumps(data, default=str)[:400]}"
    return f"{head} {msg}"


async def _run(question: str, conversation_id: str, verbose: bool, raw: bool) -> int:
    # Lazy import so --help is fast and import errors surface clearly
    from agent.orchestrator import OrchestratorAgent

    print(f"\n→ Question: {question}\n", file=sys.stderr)
    agent = OrchestratorAgent()

    in_token_stream = False
    final_event: Dict[str, Any] = {}
    error_event: Dict[str, Any] = {}
    t0 = time.time()

    async for event in agent.execute_rag_flow(
        question=question,
        conversation_id=conversation_id,
        stream=True,
    ):
        if raw:
            print(json.dumps(event, default=str))
            continue

        etype = event.get("type")

        if etype == "result":
            final_event = event
            continue
        if etype == "error":
            error_event = event
            print(f"\n\n❌ ERROR: {event.get('message') or event.get('data')}", file=sys.stderr)
            continue

        if etype == "token":
            if not in_token_stream:
                print("\n\n=== ANSWER ===\n", flush=True)
                in_token_stream = True
            sys.stdout.write(event.get("content", ""))
            sys.stdout.flush()
            continue

        # Non-token event: end any active token stream cleanly
        if in_token_stream:
            print("\n", flush=True)
            in_token_stream = False

        line = _fmt_event(event, verbose)
        if line:
            print(line, file=sys.stderr, flush=True)

    elapsed = time.time() - t0

    if error_event:
        return 1

    if final_event and not raw:
        data = final_event.get("data", {}) or {}
        resp = data.get("response", {}) or {}
        citations = resp.get("citations", []) or []
        iters = resp.get("total_iterations", 0)
        answer = resp.get("answer", "") or ""
        if not in_token_stream:
            # Token stream didn't fire (non-stream path) — print the answer now
            print("\n=== ANSWER ===\n")
            print(answer)
        print(
            f"\n\n--- {elapsed:.1f}s | {iters} tool call(s) | {len(citations)} citation(s) ---",
            file=sys.stderr,
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Debug CLI for OrchestratorAgent (deep agent).",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to ask. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging + dump full event payloads.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw event JSON, one per line (for piping to jq).",
    )
    parser.add_argument(
        "--conversation-id",
        default=f"cli-{int(time.time())}",
        help="Conversation ID for memory (default: ephemeral).",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if args.question:
        question = " ".join(args.question).strip()
    else:
        question = sys.stdin.read().strip()

    if not question:
        parser.error("no question provided (pass as args or via stdin)")

    return asyncio.run(_run(question, args.conversation_id, args.verbose, args.raw))


if __name__ == "__main__":
    sys.exit(main())
