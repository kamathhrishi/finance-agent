"""
Post-processing for datamule's HTML→markdown table output.

datamule preserves SEC's HTML structure where every dollar value occupies TWO
adjacent columns: one for the currency symbol (or empty), one for the numeric
value. The result is a markdown table where every fiscal year is duplicated:

    | Operating income | 1,493,602 | 1,493,602 | 903,095 | 903,095 | ... |
                        ^^^^^^^^^^ ^^^^^^^^^^^
                        these are the same FY2016 column

Or, with the "$" pattern preserved:

    | Subscription |   | $4,584,833 |   | $3,223,904 |   | $2,076,584 |
                   ^^ empty $-column   ^^ empty $-column   ^^ empty $-column

Either form trips up the LLM: it grabs columns 3 and 5 thinking they map to
sequential years, but they actually map to FY-1 and FY-2.

This module collapses those redundant column pairs at ingest time so the LLM
sees a clean one-column-per-year table.
"""
from __future__ import annotations

import re
from typing import List


_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_SEPARATOR_RE = re.compile(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$")
# A "currency-only" cell is empty, just whitespace, or just a $ sign.
_CURRENCY_ONLY_RE = re.compile(r"^\s*\$?\s*$")


def _split_row(line: str) -> List[str]:
    """Split a markdown-table row into cell strings (no leading/trailing edges)."""
    raw = line.strip()
    if not raw.startswith("|") or not raw.endswith("|"):
        return [line]
    # Strip the leading and trailing | then split. Cells keep internal whitespace.
    inner = raw[1:-1]
    return inner.split("|")


def _join_row(cells: List[str]) -> str:
    return "| " + " | ".join(c.strip() for c in cells) + " |"


def _is_separator_row(cells: List[str]) -> bool:
    """A markdown alignment row: |---|---|."""
    if not cells:
        return False
    return all(re.fullmatch(r"\s*:?-+:?\s*", c) for c in cells)


def _looks_currency_or_empty(cell: str) -> bool:
    return bool(_CURRENCY_ONLY_RE.match(cell))


def _strip_dollar(cell: str) -> str:
    """Remove leading whitespace + $ from a cell for content comparison."""
    return re.sub(r"^\s*\$\s*", "", cell.strip())


def _columns_equivalent(left_cells: List[str], right_cells: List[str]) -> bool:
    """
    Decide whether two adjacent columns form the SEC "$ + value" duplicate pair.

    SEC HTML uses several column patterns inside a single table:
      - row A: `| 1,235 | 1,235 |`        (value duplicated in both cells)
      - row B: `|       | $1,235 |`       (left empty, right carries $value)
      - row C: `| $897  |        |`       (left has $value, right empty)  [rare]
      - blank rows / subtotal headers

    Real, distinct adjacent columns will look NOTHING like any of these.

    The check: for every row that has SOMETHING in either cell, does it match
    one of the duplicate patterns above? If yes for ≥80% of "data" rows AND
    enough of those rows actually carried a value somewhere, the columns are
    a duplicated pair.
    """
    if len(left_cells) != len(right_cells) or not left_cells:
        return False

    n_data_rows = 0      # rows with content in at least one cell
    n_dup_pattern = 0    # rows matching one of the 3 duplicate patterns
    n_value_rows = 0     # rows where a numeric/currency value appears in either cell

    for L, R in zip(left_cells, right_cells):
        L_strip = L.strip()
        R_strip = R.strip()
        if not L_strip and not R_strip:
            continue
        n_data_rows += 1

        # Strip out trailing punctuation that datamule splits across cells
        # (e.g. negative numbers `(1,508` and `(1,508)` — left missing the close-paren)
        Ls = re.sub(r"[()$\s]+", "", L_strip)
        Rs = re.sub(r"[()$\s]+", "", R_strip)
        is_value = bool(Rs or Ls)
        if is_value:
            n_value_rows += 1

        same_after_strip = (Ls == Rs and is_value)
        left_currency_or_empty = _looks_currency_or_empty(L) or L_strip in ("(", "(  ")
        right_has_data = bool(R_strip) and not _looks_currency_or_empty(R)
        right_currency_or_empty = _looks_currency_or_empty(R) or R_strip in (")", "  )")
        left_has_data = bool(L_strip) and not _looks_currency_or_empty(L)

        # Pattern A: identical (modulo $/whitespace/parens)
        # Pattern B: left empty/$, right has value
        # Pattern C: left has value, right empty/) (datamule splits "(1,508)" into "(1,508" + ")")
        if (
            same_after_strip
            or (left_currency_or_empty and right_has_data)
            or (left_has_data and right_currency_or_empty)
        ):
            n_dup_pattern += 1

    if n_data_rows == 0 or n_value_rows < 2:
        return False
    return n_dup_pattern >= max(2, int(0.8 * n_data_rows))


def _dedupe_table_columns(rows: List[List[str]]) -> List[List[str]]:
    """
    Walk left-to-right; whenever (col i, col i+1) form a duplicate pair, drop
    col i (the $-only / empty column) and keep col i+1 (the value column).

    Operates on a parallel list of row-cells. Returns new row-cells list.
    """
    if not rows:
        return rows
    # Determine number of columns from longest row
    n_cols = max(len(r) for r in rows)
    # Pad rows to n_cols
    padded = [r + [""] * (n_cols - len(r)) for r in rows]

    keep_mask = [True] * n_cols
    i = 0
    while i + 1 < n_cols:
        if not keep_mask[i] or not keep_mask[i + 1]:
            i += 1
            continue
        left_col = [r[i] for r in padded]
        right_col = [r[i + 1] for r in padded]
        if _columns_equivalent(left_col, right_col):
            # Drop the LEFT column (it's the $-only / empty / dup column).
            # The RIGHT column carries the actual numeric value.
            # Special case: if BOTH columns have data and they're equal, drop
            # right (keep first occurrence) — but the result is the same either
            # way so we always drop left for simplicity.
            keep_mask[i] = False
            i += 2  # skip past the just-collapsed pair
        else:
            i += 1

    if all(keep_mask):
        return rows
    out: List[List[str]] = []
    for r in padded:
        out.append([r[j] for j in range(n_cols) if keep_mask[j]])
    return out


def cleanup_markdown_tables(markdown: str) -> str:
    """
    Apply table column dedupe to all markdown tables in the document.

    A "table" here is any contiguous run of lines starting and ending with `|`.
    The first row may be a header; the second may be an alignment separator.
    """
    if "|" not in markdown:
        return markdown

    out_lines: List[str] = []
    buf: List[str] = []  # current run of table-looking lines

    def flush_table():
        nonlocal buf
        if not buf:
            return
        rows = [_split_row(l) for l in buf]
        new_rows = _dedupe_table_columns(rows)
        if new_rows is rows or new_rows == rows:
            out_lines.extend(buf)
        else:
            for r in new_rows:
                out_lines.append(_join_row(r))
        buf = []

    for line in markdown.split("\n"):
        if _TABLE_LINE_RE.match(line):
            buf.append(line)
        else:
            flush_table()
            out_lines.append(line)
    flush_table()

    return "\n".join(out_lines)
