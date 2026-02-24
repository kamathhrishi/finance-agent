import bisect
import re
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1. MARKDOWN STRUCTURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_document_structure(markdown_text: str) -> List[Dict]:
    """
    Parse markdown headings into a hierarchical tree.

    Returns list of root-level nodes. Each node:
        {level: int, text: str, char_offset: int, children: [...]}
    """
    lines = markdown_text.split('\n')
    flat_headings: List[Dict] = []
    char_pos = 0
    in_fence = False

    for line in lines:
        stripped = line.rstrip()
        # Track fenced code blocks
        if stripped.startswith('```') or stripped.startswith('~~~'):
            in_fence = not in_fence
            char_pos += len(line) + 1
            continue

        if not in_fence:
            m = re.match(r'^(#{1,6})\s+(.+)', stripped)
            if m:
                level = len(m.group(1))
                text = m.group(2).strip()
                flat_headings.append({
                    'level': level,
                    'text': text,
                    'char_offset': char_pos,
                    'children': []
                })

        char_pos += len(line) + 1

    # Build tree from flat list
    root: List[Dict] = []
    stack: List[Dict] = []  # stack of (level, node)

    for node in flat_headings:
        # Pop stack until we find a parent with smaller level
        while stack and stack[-1]['level'] >= node['level']:
            stack.pop()

        if stack:
            stack[-1]['children'].append(node)
        else:
            root.append(node)

        stack.append(node)

    return root


def flatten_headings(structure: List[Dict]) -> List[Tuple[int, str, List[str]]]:
    """
    DFS flatten the heading tree to a sorted list of
    (char_offset, heading_text, full_path_list).
    """
    result: List[Tuple[int, str, List[str]]] = []

    def dfs(nodes: List[Dict], path: List[str]):
        for node in nodes:
            current_path = path + [node['text']]
            result.append((node['char_offset'], node['text'], current_path))
            dfs(node['children'], current_path)

    dfs(structure, [])
    result.sort(key=lambda x: x[0])
    return result


def extract_table_of_contents(markdown_text: str) -> List[Dict]:
    """
    Find the first markdown table that looks like a filing index (TOC).
    Returns list of {title: str, page: str}.
    """
    lines = markdown_text.split('\n')
    in_table = False
    table_rows: List[str] = []
    toc: List[Dict] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|'):
            in_table = True
            table_rows.append(stripped)
        elif in_table and table_rows:
            # Table ended — parse and return
            break

    if not table_rows:
        return []

    # Parse the first real table
    for row in table_rows:
        # Skip separator rows
        if re.match(r'^\|[\s\-|]+\|$', row):
            continue

        cells = [c.strip() for c in row.strip('|').split('|')]
        if len(cells) >= 2:
            title = cells[0]
            page = cells[-1] if cells[-1].strip().isdigit() else ''
            if title and title not in ('', 'Page', 'Item'):
                toc.append({'title': title, 'page': page})

    # Only return if it looks like an actual filing TOC (has Item 1, Item 7, etc.)
    has_items = any('Item' in e['title'] or 'PART' in e['title'].upper() for e in toc)
    return toc[:60] if has_items else []


def build_headings_map(structure: List[Dict]) -> Dict[str, int]:
    """
    Flat map: {heading_text -> char_offset} for jump-to-section.
    """
    result: Dict[str, int] = {}

    def dfs(nodes: List[Dict]):
        for node in nodes:
            result[node['text']] = node['char_offset']
            dfs(node['children'])

    dfs(structure)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. STRUCTURED CHUNK CREATION
# ─────────────────────────────────────────────────────────────────────────────

# Map heading text patterns to sec_section codes
# Use [\s\xa0]* to handle both regular spaces and non-breaking spaces
# (\xa0 is common in SEC filings: "Item\xa07." instead of "Item 7.")
_S = r'[\s\xa0]*'
SECTION_PATTERNS = [
    (r'item' + _S + r'1[aA]', 'item_1a', 'Risk Factors'),
    (r'item' + _S + r'1[bB]', 'item_1b', 'Unresolved Staff Comments'),
    (r'item' + _S + r'1\b', 'item_1', 'Business'),
    (r'item' + _S + r'2\b', 'item_2', 'Properties'),
    (r'item' + _S + r'3\b', 'item_3', 'Legal Proceedings'),
    (r'item' + _S + r'4\b', 'item_4', 'Mine Safety Disclosures'),
    (r'item' + _S + r'5\b', 'item_5', 'Market for Common Stock'),
    (r'item' + _S + r'6\b', 'item_6', 'Selected Financial Data'),
    (r'item' + _S + r'7[aA]', 'item_7a', 'Quantitative and Qualitative Disclosures'),
    (r'item' + _S + r'7\b', 'item_7', "Management's Discussion and Analysis"),
    (r'item' + _S + r'8\b', 'item_8', 'Financial Statements'),
    (r'item' + _S + r'9[aA]', 'item_9a', 'Controls and Procedures'),
    (r'item' + _S + r'9[bB]', 'item_9b', 'Other Information'),
    (r'item' + _S + r'9\b', 'item_9', 'Disagreements with Accountants'),
    (r'item' + _S + r'10\b', 'item_10', 'Directors and Executive Officers'),
    (r'item' + _S + r'11\b', 'item_11', 'Executive Compensation'),
    (r'item' + _S + r'12\b', 'item_12', 'Security Ownership'),
    (r'item' + _S + r'13\b', 'item_13', 'Certain Relationships'),
    (r'item' + _S + r'14\b', 'item_14', 'Principal Accountant Fees'),
    (r'item' + _S + r'15\b', 'item_15', 'Exhibits'),
]

_PARA_REFS = re.compile(
    r'see\s+item|as\s+described\s+in\s+item|refer\s+to\s+item|'
    r'pursuant\s+to|in\s+accordance\s+with|as\s+a\s+result\s+of|'
    r'prior\s+period|for\s+additional\s+information',
    re.IGNORECASE
)


def identify_section(heading_path: List[str]) -> Tuple[str, str]:
    """Return (sec_section, sec_section_title) from heading path.

    Skips heading elements that look like paragraph/footnote text:
    - Very long (>150 chars), OR
    - Contains cross-reference phrases like "See Item X"
    Real section headings ("Item 7. Management's Discussion...") are kept.
    """
    for heading in reversed(heading_path):
        # Skip paragraph/footnote text
        if len(heading) > 150:
            continue
        if _PARA_REFS.search(heading):
            continue
        for pattern, code, title in SECTION_PATTERNS:
            if re.search(pattern, heading, re.IGNORECASE):
                return code, title
    return 'general', 'General'


def create_structured_chunks(
    markdown_text: str,
    structure: List[Dict],
    ticker: str,
    fiscal_year: int,
    filing_type: str = '10-K',
    chunk_size: int = 1500,
    overlap: int = 200,
) -> List[Dict]:
    """
    Split markdown_text into overlapping chunks annotated with heading context.

    Each chunk dict:
        content, char_offset, parent_heading, heading_path,
        sec_section, sec_section_title, chunk_type
    """
    flat = flatten_headings(structure)
    offsets = [h[0] for h in flat]  # sorted list of heading char offsets

    # Precompute table block spans so we can keep tables intact (no splits)
    table_spans: List[Tuple[int, int]] = []
    pos = 0
    in_table = False
    table_start = 0
    for line in markdown_text.splitlines(keepends=True):
        is_table_line = line.lstrip().startswith('|')
        if is_table_line and not in_table:
            in_table = True
            table_start = pos
        if not is_table_line and in_table:
            table_spans.append((table_start, pos))
            in_table = False
        pos += len(line)
    if in_table:
        table_spans.append((table_start, pos))

    chunks: List[Dict] = []
    text_len = len(markdown_text)
    start = 0
    chunk_idx = 0
    table_idx = 0

    while start < text_len:
        # Advance table index past any spans we've already moved beyond
        while table_idx < len(table_spans) and table_spans[table_idx][1] <= start:
            table_idx += 1

        # If we're inside a table block, emit one full-table chunk
        if table_idx < len(table_spans):
            t_start, t_end = table_spans[table_idx]
            if t_start <= start < t_end:
                chunk_text = markdown_text[t_start:t_end]
                if not chunk_text.strip():
                    start = t_end
                    continue
                idx = bisect.bisect_right(offsets, t_start) - 1
                if idx >= 0:
                    parent_heading = flat[idx][1]
                    heading_path = flat[idx][2]
                else:
                    parent_heading = None
                    heading_path = []

                sec_section, sec_section_title = identify_section(heading_path)
                path_string = ' > '.join(heading_path) if heading_path else ''
                chunks.append({
                    'content': chunk_text,
                    'char_offset': t_start,
                    'parent_heading': parent_heading,
                    'heading_path': heading_path,
                    'sec_section': sec_section,
                    'sec_section_title': sec_section_title,
                    'chunk_type': 'table',
                    'path_string': path_string,
                    'chunk_index': chunk_idx,
                    'chunk_length': len(chunk_text),
                })
                start = t_end
                chunk_idx += 1
                continue

        end = min(start + chunk_size, text_len)
        # If the next table starts before our chunk end, stop before it
        if table_idx < len(table_spans):
            t_start, _t_end = table_spans[table_idx]
            if start < t_start < end:
                end = t_start

        # Don't cut inside a word
        if end < text_len:
            newline = markdown_text.rfind('\n', start, end)
            if newline > start + chunk_size // 2:
                end = newline

        chunk_text = markdown_text[start:end]
        if not chunk_text.strip():
            start = end + 1
            continue

        # Find active heading at chunk start
        idx = bisect.bisect_right(offsets, start) - 1
        if idx >= 0:
            parent_heading = flat[idx][1]
            heading_path = flat[idx][2]
        else:
            parent_heading = None
            heading_path = []

        sec_section, sec_section_title = identify_section(heading_path)

        # Detect table chunks (should be rare since tables are handled above)
        lines = chunk_text.split('\n')
        table_lines = sum(1 for l in lines if l.strip().startswith('|'))
        chunk_type = 'table' if table_lines > len(lines) * 0.3 else 'text'

        # Build path string (for backwards compat with path_string column)
        path_string = ' > '.join(heading_path) if heading_path else ''

        chunks.append({
            'content': chunk_text,
            'char_offset': start,
            'parent_heading': parent_heading,
            'heading_path': heading_path,
            'sec_section': sec_section,
            'sec_section_title': sec_section_title,
            'chunk_type': chunk_type,
            'path_string': path_string,
            'chunk_index': chunk_idx,
            'chunk_length': len(chunk_text),
        })

        # Advance with overlap
        start = end - overlap if end < text_len else text_len
        chunk_idx += 1

    return chunks


def count_nodes(structure: List[Dict]) -> int:
    """Count total nodes in heading tree."""
    count = len(structure)
    for node in structure:
        count += count_nodes(node['children'])
    return count
