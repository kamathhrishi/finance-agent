"""
Line-range highlighting for agent corpus files.

The agent emits citations as `path:line` or `path:line-line`. To render a UI
view of the cited document we read the file, wrap each cited line range in a
`<mark>` tag (the same `mark.highlighted-chunk` class the SECFilingViewer
already styles), and return the marked-up markdown.

Tables in datamule-extracted markdown are rows like `| col | col |\n`. Wrapping
the whole row in `<mark>...</mark>` keeps the row a valid table row (the
markdown renderer parses `|`-delimited cells from inside the mark; rehype-raw
then preserves the mark as inline markup). For headings (`## Item 7. ...`),
we insert the open tag *after* the `#` prefix so heading parsing is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import re


@dataclass
class LineHighlight:
    chunk_id: str
    line_start: int  # 1-based, inclusive
    line_end: int    # 1-based, inclusive
    primary: bool = False  # marks the citation that was clicked


_HEADING_PREFIX = re.compile(r"^(\s*#{1,6}\s+)")


def _wrap_line(line: str, chunk_id: str, primary: bool) -> str:
    """Wrap a single source line in a <mark>, preserving heading/table syntax."""
    cls = "highlighted-chunk highlighted-chunk-primary" if primary else "highlighted-chunk"
    open_tag = f'<mark class="{cls}" data-chunk-id="{chunk_id}">'
    close_tag = "</mark>"

    if not line.strip():
        return line

    m = _HEADING_PREFIX.match(line)
    if m:
        prefix_end = m.end()
        return line[:prefix_end] + open_tag + line[prefix_end:] + close_tag

    return open_tag + line + close_tag


def inject_line_highlights(markdown: str, highlights: Sequence[LineHighlight]) -> str:
    """
    Wrap each cited line range in `<mark>`. Idempotent for non-overlapping ranges.

    Overlapping ranges: later highlights win (we process in order; if a line
    was already wrapped by a prior highlight, we leave it alone).
    """
    if not highlights:
        return markdown

    lines = markdown.split("\n")
    n = len(lines)
    wrapped = [False] * n

    for h in highlights:
        # Convert to 0-based, clamp into range
        start = max(0, h.line_start - 1)
        end = min(n - 1, h.line_end - 1)
        if start > end:
            continue
        for i in range(start, end + 1):
            if wrapped[i]:
                continue
            lines[i] = _wrap_line(lines[i], h.chunk_id, h.primary)
            wrapped[i] = True

    return "\n".join(lines)


def extract_snippet(markdown: str, line_start: int, line_end: int, max_chars: int = 280) -> str:
    """Pull a short text snippet from the file (for citation cards in the UI)."""
    lines = markdown.split("\n")
    start = max(0, line_start - 1)
    end = min(len(lines), line_end)
    snippet = " ".join(s.strip() for s in lines[start:end] if s.strip())
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 1].rstrip() + "…"
    return snippet
