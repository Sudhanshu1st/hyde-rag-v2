"""Structure-first chunking with stable IDs (doc_id:page:chunk) and char spans.

* Small neighbouring sections are merged until a chunk reaches TARGET_WORDS; a chapter-like
  heading (level <= 2) starts a new chunk once the current one has MIN_WORDS.
* Long sections are split at line boundaries into pieces of at most MAX_WORDS
  (about 512 embedder tokens).
* `chunk` in the ID counts chunks starting on the same page, so IDs stay stable when other
  pages change. Each chunk keeps its section path, used as a header when indexing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from hyde_rag.ingest.pdf import ParsedDoc

MAX_WORDS = 350
TARGET_WORDS = 200
MIN_WORDS = 60  # don't close a chunk at a major heading below this size


@dataclass(frozen=True)
class DocChunk:
    chunk_id: str
    doc_id: str
    page: int
    chunk: int
    section: str  # heading path of the chunk's first section, joined with " > "
    text: str  # chunk body: an exact slice of the canonical document text
    char_start: int
    char_end: int

    def as_dict(self) -> dict:
        return asdict(self)

    def index_text(self, title: str) -> str:
        """Text used for BM25 and embeddings: document title and section path as a header."""
        header = f"{title} | {self.section}" if self.section else title
        return f"{header}\n{self.text}"


def chunk_document(
    doc: ParsedDoc, max_words: int = MAX_WORDS, target_words: int = TARGET_WORDS
) -> list[DocChunk]:
    offsets = iter(doc.line_offsets)
    pieces: list[tuple[str, int, int]] = []  # (section path, start, end)
    buf: list[tuple[int, int]] = []
    buf_path, buf_words = "", 0

    def flush() -> None:
        nonlocal buf, buf_words
        if buf:
            pieces.append((buf_path, buf[0][0], buf[-1][1]))
        buf, buf_words = [], 0

    for section in doc.sections:
        path = " > ".join(section.path)
        spans = [next(offsets) for _ in section.lines]
        words = [len(line.text.split()) for line in section.lines]
        major = section.level <= 2 and buf_words >= MIN_WORDS
        if major or (buf and buf_words + sum(words) > max_words):
            flush()
        if not buf:
            buf_path = path
        for span, n in zip(spans, words, strict=True):
            if buf and buf_words + n > max_words:  # a long section: split at line boundaries
                flush()
                buf_path = path
            buf.append(span)
            buf_words += n
        if buf_words >= target_words:
            flush()
    flush()

    chunks: list[DocChunk] = []
    per_page: dict[int, int] = {}
    for path, start, end in pieces:
        page = doc.page_at(start)
        idx = per_page.get(page, 0)
        per_page[page] = idx + 1
        chunks.append(
            DocChunk(
                f"{doc.doc_id}:{page}:{idx}", doc.doc_id, page, idx, path, doc.text[start:end], start, end
            )
        )
    return chunks


def overlaps(chunk: DocChunk, doc_id: str, start: int, end: int) -> bool:
    """Gold evidence is a char span; a chunk is relevant if it overlaps it."""
    return chunk.doc_id == doc_id and chunk.char_start < end and start < chunk.char_end
