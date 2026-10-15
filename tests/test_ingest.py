"""Parser and chunker on synthetic page lines (no PDF needed)."""

from hyde_rag.ingest import pdf
from hyde_rag.ingest.chunking import chunk_document, overlaps
from hyde_rag.ingest.pdf import Line, drop_running_lines, parse_pdf

H = 792.0  # page height


def body(text, page, top, size=11.0, bold=False):
    return Line(text, page, size, bold, top, H)


def make_doc():
    lines = []
    for p in range(1, 7):
        lines.append(body("Contains Nonbinding Recommendations", p, 40, bold=True))  # running header
        lines.append(body(f"Page {p} of 6", p, 760))  # page number footer
    # Page 1: title + table of contents (entries recur later as real headings)
    lines += [
        body("Chapter 3: Cooling Requirements", 1, 100, size=16, bold=True),
        body("Contents", 1, 140, bold=True),
        body("3.1 Cooling Limits", 1, 160, bold=True),
        body("3.2 Monitoring", 1, 180, bold=True),
    ]
    # Pages 2-5: real sections
    lines += [
        body("3.1 Cooling Limits", 2, 100, bold=True),
        body("Cooked food shall be cooled from 57C to 21C within 2 hours", 2, 130),
        body("and from 21C to 5C within a total of 6 hours.", 2, 145),
        body("3.1.1 Shellstock and", 3, 100, bold=True),
        body("Shell Eggs", 3, 115, bold=True),  # heading wrapping onto a second line
        body("Shellstock received at 7C shall be cooled to 5C within 4 hours.", 3, 140),
        body("3.2 Monitoring", 4, 100, bold=True),
    ]
    lines += [
        body(" ".join(["Record the temperature at each step"] * 6) + ".", 4, 130 + 15 * i) for i in range(12)
    ]
    lines.append(body("Corrective actions follow a deviation.", 5, 100))
    return lines


def test_running_headers_and_page_numbers_are_dropped():
    kept = drop_running_lines(make_doc())
    texts = [ln.text for ln in kept]
    assert "Contains Nonbinding Recommendations" not in texts
    assert not any(t.startswith("Page ") for t in texts)


def test_sections_wrapping_toc_and_nesting(monkeypatch):
    monkeypatch.setattr(pdf, "extract_lines", lambda path: make_doc())
    doc = parse_pdf("unused.pdf", "fc", "Food Code")
    paths = [" > ".join(s.path) for s in doc.sections]
    # TOC entries on page 1 are not headings; the wrapped heading is one heading; 3.1.1 nests under 3.1.
    assert paths.count("Chapter 3: Cooling Requirements > 3.1 Cooling Limits") == 1
    assert "Chapter 3: Cooling Requirements > 3.1 Cooling Limits > 3.1.1 Shellstock and Shell Eggs" in paths
    assert "Chapter 3: Cooling Requirements > 3.2 Monitoring" in paths
    assert doc.page_at(doc.text.index("Shellstock received")) == 3


def test_chunks_are_exact_spans_with_stable_ids(monkeypatch):
    monkeypatch.setattr(pdf, "extract_lines", lambda path: make_doc())
    doc = parse_pdf("unused.pdf", "fc", "Food Code")
    chunks = chunk_document(doc, max_words=120, target_words=60)
    for c in chunks:
        assert doc.text[c.char_start : c.char_end] == c.text
        assert c.chunk_id == f"fc:{c.page}:{c.chunk}"
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    # The long monitoring section is split, and no chunk exceeds the cap (+1 line).
    monitoring = [c for c in chunks if c.section.endswith("3.2 Monitoring")]
    assert len(monitoring) >= 2 and all(len(c.text.split()) <= 120 + 40 for c in chunks)
    # Gold evidence is a char span: the cooling-limit sentence maps to exactly one chunk.
    start = doc.text.index("Cooked food shall be cooled")
    end = doc.text.index("6 hours.") + len("6 hours.")
    hits = [c.chunk_id for c in chunks if overlaps(c, "fc", start, end)]
    assert len(hits) == 1
    assert chunks[0].index_text("Food Code").startswith("Food Code | ")
