"""Retrieved chunks with stable IDs (doc_id:page:chunk) and the context block shown to the LLM."""

from __future__ import annotations

from dataclasses import dataclass


def make_chunk_id(doc_id: str, page: int, chunk: int) -> str:
    return f"{doc_id}:{page}:{chunk}"


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    page: int  # 0 when the source has no pages (e.g. a BEIR abstract)
    chunk: int
    text: str
    title: str = ""
    section: str = ""
    score: float | None = None  # retrieval or rerank score

    @property
    def chunk_id(self) -> str:
        return make_chunk_id(self.doc_id, self.page, self.chunk)


def format_context(chunks: list[Chunk]) -> str:
    """One block per chunk, headed by its ID so the model can cite it."""
    blocks = []
    for c in chunks:
        header = f"[{c.chunk_id}]"
        label = " | ".join(x for x in (c.title, c.section) if x)
        if label:
            header += f" {label}"
        blocks.append(f"{header}\n{c.text.strip()}")
    return "\n\n".join(blocks)
