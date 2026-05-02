#!/usr/bin/env python3
"""
Filesystem tools exposed to the agent.

Four tools, all sandboxed to a single root directory (typically `data/`):
    - ls(path)
    - read_file(path, offset, limit)
    - grep(pattern, path, glob, context, max_results, ignore_case)
    - glob(pattern)

Path safety: every input path is resolved relative to the sandbox root and
checked via os.path.commonpath against the realpath of the root. Any attempt
to escape (e.g. `../../etc/passwd`) raises ValueError.

`grep` shells out to `rg` (ripgrep) for speed. Ripgrep is a hard dependency.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("fs_research_agent.tools")


class SandboxViolation(ValueError):
    """Raised when a tool input attempts to escape the sandbox."""


# ─────────────────────────────────────────────────────────────────────────────
# Sandbox helper
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Sandbox:
    """Holds the root directory and resolves user-supplied paths safely."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Sandbox root does not exist or is not a dir: {self.root}")

    def resolve(self, user_path: str) -> Path:
        """Resolve `user_path` (relative to root, or absolute under root)."""
        if not user_path or user_path == ".":
            return self.root
        # Treat absolute paths as relative to the sandbox if they happen to
        # share the prefix; otherwise reject. Most agent inputs will be relative.
        p = Path(user_path)
        if p.is_absolute():
            candidate = p.resolve()
        else:
            candidate = (self.root / p).resolve()
        try:
            common = os.path.commonpath([str(candidate), str(self.root)])
        except ValueError:
            raise SandboxViolation(f"Path '{user_path}' is outside the sandbox.")
        if common != str(self.root):
            raise SandboxViolation(f"Path '{user_path}' is outside the sandbox.")
        return candidate

    def rel(self, p: Path) -> str:
        """Return a path relative to the sandbox root, for display."""
        try:
            return str(Path(p).resolve().relative_to(self.root))
        except ValueError:
            return str(p)


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────


def tool_ls(sandbox: Sandbox, path: str = ".") -> str:
    target = sandbox.resolve(path)
    if not target.exists():
        return f"ls: {path}: no such file or directory"
    if target.is_file():
        size = target.stat().st_size
        return f"FILE  {sandbox.rel(target)}  ({size:,} bytes)"
    entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    if not entries:
        return f"(empty directory: {sandbox.rel(target)})"
    lines = [f"# {sandbox.rel(target) or '.'}"]
    for e in entries:
        rel = sandbox.rel(e)
        if e.is_dir():
            n = sum(1 for _ in e.iterdir())
            lines.append(f"  DIR   {rel}/  ({n} entr{'y' if n == 1 else 'ies'})")
        else:
            lines.append(f"  FILE  {rel}  ({e.stat().st_size:,} bytes)")
    return "\n".join(lines)


def tool_read_file(
    sandbox: Sandbox, path: str, offset: int = 0, limit: int = 400
) -> str:
    target = sandbox.resolve(path)
    if not target.exists():
        return f"read_file: {path}: no such file"
    if target.is_dir():
        return f"read_file: {path}: is a directory (use ls)"
    # Cap limit to keep tool responses bounded
    limit = max(1, min(int(limit), 1500))
    offset = max(0, int(offset))

    with open(target, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    total = len(lines)
    end = min(offset + limit, total)
    chunk = lines[offset:end]

    rel = sandbox.rel(target)
    header = f"=== {rel}  (lines {offset + 1}-{end} of {total}) ==="
    body = "".join(f"{i + offset + 1:>6}  {ln.rstrip()}" + "\n" for i, ln in enumerate(chunk))
    footer = ""
    if end < total:
        footer = f"\n--- (truncated; {total - end} more lines — call again with offset={end}) ---"
    return f"{header}\n{body}{footer}"


def tool_grep(
    sandbox: Sandbox,
    pattern: str,
    path: str = ".",
    glob: Optional[str] = None,
    context: int = 2,
    max_results: int = 50,
    ignore_case: bool = True,
) -> str:
    target = sandbox.resolve(path)
    if not target.exists():
        return f"grep: {path}: no such file or directory"

    rg = shutil.which("rg")
    if not rg:
        return "grep: ripgrep (rg) is not installed; cannot run grep"

    cmd: List[str] = [rg, "--line-number", "--no-heading", "--color", "never"]
    if ignore_case:
        cmd.append("-i")
    cmd += ["-C", str(max(0, min(int(context), 5)))]
    cmd += ["--max-count", str(max(1, min(int(max_results), 200)))]
    if glob:
        cmd += ["--glob", glob]
    cmd += ["-e", pattern, str(target)]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(sandbox.root),
        )
    except subprocess.TimeoutExpired:
        return "grep: timed out after 30s"

    if proc.returncode not in (0, 1):  # 1 = no matches; 2+ = error
        err = proc.stderr.strip() or proc.stdout.strip()
        return f"grep: ripgrep error: {err}"

    out = proc.stdout
    if not out.strip():
        return f"grep: no matches for {pattern!r} in {path}"

    # Strip absolute path prefix so output is relative
    root_str = str(sandbox.root) + os.sep
    cleaned_lines: List[str] = []
    for ln in out.splitlines():
        if ln.startswith(root_str):
            cleaned_lines.append(ln[len(root_str):])
        else:
            cleaned_lines.append(ln)

    # Cap response size hard at ~12k chars to keep token use sane
    response = "\n".join(cleaned_lines)
    if len(response) > 12000:
        response = response[:12000] + "\n... (truncated; refine pattern or use --glob)"
    return response


def tool_glob(sandbox: Sandbox, pattern: str) -> str:
    """Use Path.glob (supports ** for recursive). Pattern is relative to root."""
    if not pattern:
        return "glob: pattern is required"
    matches: List[Path] = list(sandbox.root.glob(pattern))
    if not matches:
        return f"glob: no matches for {pattern!r}"
    matches.sort()
    out_lines: List[str] = []
    for m in matches[:200]:
        rel = sandbox.rel(m)
        if m.is_dir():
            out_lines.append(f"DIR   {rel}/")
        else:
            try:
                out_lines.append(f"FILE  {rel}  ({m.stat().st_size:,} bytes)")
            except OSError:
                out_lines.append(f"FILE  {rel}")
    if len(matches) > 200:
        out_lines.append(f"... ({len(matches) - 200} more)")
    return "\n".join(out_lines)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI tool schemas
# ─────────────────────────────────────────────────────────────────────────────


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": (
                "List the contents of a directory inside the research corpus. "
                "Pass `.` (or omit) to list the root."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the corpus root. Defaults to `.`",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file with line numbers. For large files use `offset` "
                "and `limit` to paginate (limit defaults to 400, max 1500)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to corpus root."},
                    "offset": {"type": "integer", "description": "First line index (0-based)."},
                    "limit": {"type": "integer", "description": "Max lines to return (default 400, max 1500)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search file contents using ripgrep. Returns matching lines "
                "with `path:line: text` and surrounding context. Use `glob` "
                "to scope to a subset of files (e.g. `**/item-7-*.md`). "
                "Case-insensitive by default."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern (ripgrep syntax)."},
                    "path": {"type": "string", "description": "Directory or file to search. Defaults to `.`"},
                    "glob": {"type": "string", "description": "Optional glob to scope files (e.g. `**/item-1a-*.md`)."},
                    "context": {"type": "integer", "description": "Lines of context around each match (default 2, max 5)."},
                    "max_results": {"type": "integer", "description": "Max matches per file (default 50, max 200)."},
                    "ignore_case": {"type": "boolean", "description": "Case-insensitive match (default true)."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": (
                "Find files by glob pattern, e.g. `filings/*/10-K/*/sections/item-7-*.md`. "
                "Supports `**` for recursion."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern relative to corpus root."},
                },
                "required": ["pattern"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────


def make_tool_executor(sandbox: Sandbox) -> Callable[[str, Dict[str, Any]], str]:
    """Return a function that runs a tool call by name and returns string output."""

    def execute(name: str, args: Dict[str, Any]) -> str:
        try:
            if name == "ls":
                return tool_ls(sandbox, args.get("path", "."))
            if name == "read_file":
                return tool_read_file(
                    sandbox,
                    path=args["path"],
                    offset=args.get("offset", 0),
                    limit=args.get("limit", 400),
                )
            if name == "grep":
                return tool_grep(
                    sandbox,
                    pattern=args["pattern"],
                    path=args.get("path", "."),
                    glob=args.get("glob"),
                    context=args.get("context", 2),
                    max_results=args.get("max_results", 50),
                    ignore_case=args.get("ignore_case", True),
                )
            if name == "glob":
                return tool_glob(sandbox, pattern=args["pattern"])
            return f"error: unknown tool {name!r}"
        except SandboxViolation as e:
            return f"error: {e}"
        except KeyError as e:
            return f"error: missing required argument {e}"
        except Exception as e:
            logger.exception(f"tool {name} failed")
            return f"error: tool {name} raised {type(e).__name__}: {e}"

    return execute
