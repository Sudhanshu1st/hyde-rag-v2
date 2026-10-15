"""Structure-aware PDF ingestion for regulatory documents (pdfplumber, MIT).

Pipeline per document:
  1. lines with font size / boldness (pdfplumber text lines + chars);
  2. drop page numbers, table-of-contents leader lines, and running headers/footers;
  3. detect headings: larger-than-body font, short bold lines, or regulatory numbering
     (e.g. "3-501.14 Cooling.", "§ 117.135", "Chapter 5", "PART II", "5.2.5"); headings that
     wrap onto a second line are merged; nesting uses numbering depth and font size;
  4. canonical document text = surviving lines joined by "\\n", with page start offsets.
Chunking (ingest/chunking.py) works on the sections found here.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

NUMBERED_HEADING = re.compile(
    r"^(?:"
    r"(?:chapter|part|subpart|annex|appendix|section|schedule)\s+[\dIVXLA-Z]+\b"  # Chapter 3, PART II
    r"|\N{SECTION SIGN}\s*\d+"  # § 117.135
    r"|\d-\d{3}(?:\.\d+)?\s+[A-Z]"  # Food Code: 3-501.14 Cooling
    r"|\d+(?:\.\d+){1,3}\.?\s+[A-Z]"  # 5.2.5 Title
    r"|[IVX]+\.\s+[A-Z]"  # IV. Title
    r"|[A-H]\.\s+[A-Z][a-z]"  # B. Title
    r")",
    re.IGNORECASE,
)
PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d{1,4}(?:\s+of\s+\d{1,4})?$|^[ivxlc]{1,6}$", re.IGNORECASE)
TOC_LEADER = re.compile(r"(?:\.\s?){4,}\s*\d*\s*$|\N{HORIZONTAL ELLIPSIS}{2,}")
_DEPTH = re.compile(r"^(\d+(?:\.\d+)+)")
_WS = re.compile(r"\s+")


@dataclass
class Line:
    text: str
    page: int  # 1-based
    size: float
    bold: bool
    top: float
    page_height: float


@dataclass
class Section:
    heading: str  # "" for text before the first heading
    path: list[str]  # heading hierarchy, outermost first
    page: int  # page where the section starts
    lines: list[Line] = field(default_factory=list)
    level: int = 4  # heading level (1 = chapter-like)


@dataclass
class ParsedDoc:
    doc_id: str
    title: str
    text: str  # canonical text; gold spans and chunk offsets index into this
    page_starts: list[int]  # char offset where each page starts (index i -> page i + 1)
    sections: list[Section]
    line_offsets: list[tuple[int, int]]  # (start, end) in text for every kept line, in order

    def page_at(self, offset: int) -> int:
        page = 1
        for i, start in enumerate(self.page_starts):
            if start <= offset:
                page = i + 1
        return page


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _key(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower())


def find_gutter(words: list[dict], x_lo: float, x_hi: float) -> float | None:
    """x of a vertical gutter between x_lo and x_hi, or None.

    Column layouts and multi-column tables leave a vertical band with (almost) no words.
    A few words may cross it (e.g. "OR" markers or a full-width heading), so a band counts as
    a gutter when its word coverage is under 10% of the typical coverage elsewhere, it lies in
    the middle 40% of the region, it is at least 4 pt wide, and both sides hold >= 20% of words.
    """
    if len(words) < 40:
        return None
    lo, hi = int(x_lo), int(x_hi) + 1
    occ = [0] * (hi - lo + 1)
    for w in words:
        for x in range(max(lo, int(w["x0"])), min(hi, int(w["x1"])) + 1):
            occ[x - lo] += 1
    busy = sorted(v for v in occ if v > 0)
    if not busy:
        return None
    threshold = max(1, 0.1 * busy[len(busy) // 2])
    span = hi - lo
    start_search, end_search = int(0.3 * span), int(0.7 * span)
    best: tuple[int, int] | None = None
    run_start = None
    for i in range(start_search, end_search + 1):
        low = occ[i] <= threshold
        if low and run_start is None:
            run_start = i
        if (not low or i == end_search) and run_start is not None:
            end = i if not low else i + 1
            if end - run_start >= 4 and (best is None or end - run_start > best[1] - best[0]):
                best = (run_start, end)
            run_start = None
    if best is None:
        return None
    gutter = lo + (best[0] + best[1]) / 2
    left = sum(1 for w in words if (w["x0"] + w["x1"]) / 2 < gutter)
    if min(left, len(words) - left) < 0.2 * len(words):
        return None
    return gutter


def _column_regions(page, depth: int = 2) -> list:
    """Split a page into column regions (left to right), recursively for 3-4 columns."""

    def split(region, x0: float, x1: float, level: int) -> list:
        words = region.extract_words()
        g = find_gutter(words, x0, x1) if level < depth else None
        if g is None:
            return [region]
        left = region.filter(lambda o, g=g: o.get("object_type") != "char" or (o["x0"] + o["x1"]) / 2 < g)
        right = region.filter(lambda o, g=g: o.get("object_type") != "char" or (o["x0"] + o["x1"]) / 2 >= g)
        return split(left, x0, g, level + 1) + split(right, g, x1, level + 1)

    return split(page, 0.0, float(page.width), 0)


def extract_lines(pdf_path: Path) -> list[Line]:
    import pdfplumber

    lines: list[Line] = []
    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            # Multi-column pages are read column by column instead of straight across.
            for region in _column_regions(page):
                for tl in region.extract_text_lines(return_chars=True, strip=True):
                    text = _norm(tl["text"])
                    chars = [c for c in tl.get("chars", []) if c.get("text", "").strip()]
                    if not text or not chars:
                        continue
                    size = statistics.median(c.get("size", 0.0) for c in chars)
                    bold = sum("bold" in str(c.get("fontname", "")).lower() for c in chars) > len(chars) / 2
                    lines.append(Line(text, pno, round(size, 1), bold, float(tl["top"]), float(page.height)))
    return lines


def drop_running_lines(lines: list[Line]) -> list[Line]:
    """Remove page numbers, table-of-contents leader lines, and running headers/footers.

    A running line is short and repeats on many pages: on >= 30% of pages when it sits in the
    top/bottom 12% of the page, or on >= 60% of pages anywhere.
    """
    n_pages = len({ln.page for ln in lines})
    pages_with: dict[str, set[int]] = {}
    for ln in lines:
        if len(ln.text.split()) <= 12:
            pages_with.setdefault(_key(ln.text), set()).add(ln.page)
    kept = []
    for ln in lines:
        at_edge = ln.top < 0.12 * ln.page_height or ln.top > 0.88 * ln.page_height
        share = len(pages_with.get(_key(ln.text), ())) / n_pages if n_pages >= 4 else 0.0
        if at_edge and PAGE_NUMBER.match(ln.text):
            continue
        if (at_edge and share >= 0.3) or share >= 0.6:
            continue
        if TOC_LEADER.search(ln.text):
            continue
        kept.append(ln)
    return kept


def body_size(lines: list[Line]) -> float:
    weights: dict[float, int] = {}
    for ln in lines:
        weights[ln.size] = weights.get(ln.size, 0) + len(ln.text)
    return max(weights, key=weights.__getitem__) if weights else 10.0


def is_heading(line: Line, body: float) -> bool:
    text = line.text
    if len(text) > 140 or len(text) < 3 or text.endswith((",", ";")):
        return False
    if not (text[0].isupper() or text[0].isdigit() or text[0] in "(§"):
        return False  # body text continuing from the previous line
    if line.size >= body + 1.0 and len(text.split()) <= 20:
        return True
    if NUMBERED_HEADING.match(text) and len(text.split()) <= 18:
        return True
    return line.bold and len(text.split()) <= 14 and not text.endswith(".")


def heading_level(line: Line, body: float) -> int:
    """Smaller = more major. Combines numbering depth and font size (approximate, per doc)."""
    text = line.text
    if re.match(r"^(chapter|part|annex|appendix|schedule)\b", text, re.IGNORECASE):
        return 1
    m = _DEPTH.match(text)
    if m:
        return 1 + m.group(1).count(".")  # 11.1 -> 2, 11.1.3 -> 3
    if re.match(r"^\d-\d{3}", text) or text.startswith("\N{SECTION SIGN}"):
        return 3
    if re.match(r"^[IVX]+\.\s", text):
        return 2
    if re.match(r"^[A-H]\.\s", text):
        return 3
    if line.size >= body + 3.0:
        return 1
    if line.size >= body + 1.0:
        return 2
    return 4


def toc_lines(lines: list[Line], body: float) -> set[int]:
    """ids of table-of-contents lines: "Contents" titles, and heading-like lines in the first
    pages whose text appears again further on (that later occurrence is the real heading).
    Only the first pages are eligible, so headings legitimately repeated in the body are kept."""
    n_pages = max((ln.page for ln in lines), default=1)
    early = max(3, n_pages // 10)
    last_index: dict[str, int] = {}
    for i, ln in enumerate(lines):
        last_index[_key(ln.text)] = i
    out = set()
    for i, ln in enumerate(lines):
        if ln.page > early:
            break
        title = ln.text.lower().strip(" :") in ("contents", "table of contents")
        if title or (is_heading(ln, body) and last_index[_key(ln.text)] > i):
            out.add(id(ln))
    return out


def _wraps(prev: Line | None, prev_was_heading: bool, ln: Line) -> bool:
    """True if `ln` continues the heading on the previous line."""
    return (
        prev is not None
        and prev_was_heading
        and prev.page == ln.page
        and prev.size == ln.size
        and prev.bold == ln.bold
        and 0 < ln.top - prev.top < 2.2 * max(ln.size, 1.0)
        and not prev.text.endswith((".", ":"))
        and not NUMBERED_HEADING.match(ln.text)
    )


def parse_pdf(pdf_path: Path, doc_id: str, title: str) -> ParsedDoc:
    lines = drop_running_lines(extract_lines(pdf_path))
    body = body_size(lines)
    toc = toc_lines(lines, body)

    text_parts: list[str] = []
    page_starts: list[int] = []
    line_offsets: list[tuple[int, int]] = []
    sections: list[Section] = [Section("", [], lines[0].page if lines else 1, level=99)]
    stack: list[tuple[int, str]] = []  # (level, heading text)
    cursor, current_page = 0, 0
    prev: Line | None = None
    prev_was_heading = False
    for ln in lines:
        while current_page < ln.page:  # record page starts, including empty pages
            page_starts.append(cursor)
            current_page += 1
        start = cursor
        text_parts.append(ln.text)
        cursor += len(ln.text) + 1  # "\n" separator
        line_offsets.append((start, start + len(ln.text)))
        heading = id(ln) not in toc and is_heading(ln, body)
        if heading and stack and _wraps(prev, prev_was_heading, ln):
            sec = sections[-1]
            sec.heading = f"{sec.heading} {ln.text}"
            sec.path[-1] = sec.heading
            stack[-1] = (stack[-1][0], sec.heading)
            sec.lines.append(ln)
        elif heading:
            level = heading_level(ln, body)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, ln.text))
            sections.append(Section(ln.text, [h for _, h in stack], ln.page, [ln], level=level))
        else:
            sections[-1].lines.append(ln)
        prev, prev_was_heading = ln, heading
    return ParsedDoc(
        doc_id=doc_id,
        title=title,
        text="\n".join(text_parts),
        page_starts=page_starts,
        sections=[s for s in sections if s.lines],
        line_offsets=line_offsets,
    )
