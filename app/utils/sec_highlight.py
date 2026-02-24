import re
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

MIN_HIGHLIGHT_CHARS = 1500  # Always show at least this much context
LOOK_AHEAD = 600           # Extra chars after each chunk end to catch adjacent headings


def _mark_region(text: str, chunk_id: str) -> str:
    """
    Wrap each non-blank LINE with its own <mark> tag so that:
    - Single-newline separators work (not just double-newline paragraph breaks)
    - Heading lines get the mark inserted AFTER the ## prefix so remark-gfm
      still recognises them as headings: ## <mark>Heading text</mark>
    - Table rows are left unmarked to preserve table pipe syntax
    - Blank lines are preserved as-is
    """
    mark_open = f'<mark class="highlighted-chunk" data-chunk-id="{chunk_id}">'
    mark_close = '</mark>'

    result = []
    for line in text.split('\n'):
        if not line.strip():
            result.append(line)
            continue

        # Markdown heading: put mark inside the heading after ## prefix
        heading_match = re.match(r'^(#{1,6}\s+)', line)
        if heading_match:
            prefix = heading_match.group(1)
            result.append(prefix + mark_open + line[len(prefix):] + mark_close)
        # Table separator row (|---|---|): leave unchanged
        elif re.match(r'^[\s|:\-=]+$', line) and '|' in line:
            result.append(line)
        # Table data row: mark content inside each cell to preserve pipe syntax
        elif line.lstrip().startswith('|'):
            parts = line.split('|')
            marked_parts = [
                mark_open + p + mark_close if p.strip() else p
                for p in parts
            ]
            result.append('|'.join(marked_parts))
        else:
            result.append(mark_open + line + mark_close)

    return '\n'.join(result)


def inject_sec_highlights(markdown: str, chunks: List[Dict]) -> str:
    if not markdown or not chunks:
        return markdown
    result = markdown
    spans: List[Tuple[int, int, str]] = []

    for i, chunk in enumerate(chunks):
        text = chunk.get("chunk_text") or ""
        char_offset = chunk.get("char_offset")
        chunk_length = chunk.get("chunk_length")
        chunk_id = chunk.get("chunk_id") or chunk.get("chunk_index") or f"chunk-{i}"

        logger.info(f"[HIGHLIGHT] chunk {i}: id={chunk_id} char_offset={char_offset} chunk_length={chunk_length} text_len={len(text)} text_preview={text[:60]!r}")

        if isinstance(char_offset, int) and isinstance(chunk_length, int) and chunk_length > 0:
            # Extend to MIN_HIGHLIGHT_CHARS so short boundary-chunks still show
            # enough context; add LOOK_AHEAD to catch section headings that
            # immediately follow a chunk boundary
            effective_length = max(chunk_length, MIN_HIGHLIGHT_CHARS)
            end = min(char_offset + effective_length + LOOK_AHEAD, len(result))
            if 0 <= char_offset < len(result):
                logger.info(f"[HIGHLIGHT] chunk {i}: PRIMARY path span=({char_offset}, {end}) effective_length={effective_length}")
                spans.append((char_offset, end, str(chunk_id)))
                continue

        # Fallback: anchor on first ~150 chars of chunk text, then extend window
        if text and len(text) >= 20:
            anchor = text.strip()[:150]
            idx = result.find(anchor)
            if idx != -1:
                end = min(idx + max(len(text), MIN_HIGHLIGHT_CHARS), len(result))
                logger.info(f"[HIGHLIGHT] chunk {i}: FALLBACK exact span=({idx}, {end})")
                spans.append((idx, end, str(chunk_id)))
                continue

            # Flexible whitespace match
            tokens = re.split(r'\s+', anchor)
            if len(tokens) >= 3:
                pattern = r'\s+'.join(re.escape(t) for t in tokens if t)
                m = re.search(pattern, result)
                if m:
                    start = m.start()
                    end = min(start + max(len(text), MIN_HIGHLIGHT_CHARS), len(result))
                    logger.info(f"[HIGHLIGHT] chunk {i}: FALLBACK fuzzy span=({start}, {end})")
                    spans.append((start, end, str(chunk_id)))
                    continue

        logger.warning(f"[HIGHLIGHT] chunk {i}: NO MATCH FOUND - chunk_id={chunk_id}")

    # Insert marks in reverse order to preserve offsets.
    # Trim (don't skip) spans that overlap with an already-marked region so that
    # closely-spaced chunks don't get silently dropped.
    spans.sort(key=lambda x: x[0], reverse=True)
    last_start = len(result) + 1
    for start, end, chunk_id in spans:
        if start >= last_start:
            continue  # completely inside an already-marked region
        if end > last_start:
            end = last_start  # trim to avoid overlap
        if end <= start:
            continue
        region = result[start:end]
        marked = _mark_region(region, chunk_id)
        mark_count = marked.count('<mark')
        logger.info(f"[HIGHLIGHT] chunk region lines={region.count(chr(10))+1} marks_produced={mark_count} first_line={repr(region.split(chr(10))[0][:60])}")
        result = result[:start] + marked + result[end:]
        last_start = start

    return result
