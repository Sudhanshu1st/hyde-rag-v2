"""Build the food-safety corpus: PDFs in data/food/raw*/ -> canonical text + chunks.

Outputs (data/food/processed/, gitignored because it includes restricted documents):
  text/<doc_id>.txt   canonical text; gold evidence spans in the eval set index into it
  docs.jsonl          one record per document (metadata, pages, page start offsets, sha256)
  chunks.jsonl        one record per chunk (chunk_id, section, page, char span, text)
Parsing is deterministic for a given PDF (sha256-pinned in data_sources.yaml) and pinned
pdfplumber version, so anyone rebuilding gets the same offsets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from hyde_rag import paths
from hyde_rag.ingest.chunking import DocChunk, chunk_document
from hyde_rag.ingest.pdf import parse_pdf

MIN_CHARS_PER_PAGE = 200  # below this the PDF is probably scanned (no text layer)
# Bump when parsing or chunking changes, so cached documents are re-parsed.
PARSER_VERSION = 2  # 2: column-aware extraction


def food_dir() -> Path:
    return paths.home() / "data" / "food"


def processed_dir() -> Path:
    return food_dir() / "processed"


def load_sources() -> list[dict]:
    return yaml.safe_load((paths.REPO_ROOT / "data_sources.yaml").read_text(encoding="utf-8"))["documents"]


def pdf_path(doc: dict) -> Path:
    sub = "raw" if doc["commit_raw"] else "raw_restricted"
    return food_dir() / sub / f"{doc['doc_id']}.pdf"


@dataclass
class IngestReport:
    parsed: list[str]
    reused: list[str]
    missing: list[str]
    low_text: list[str]
    n_chunks: int


def ingest(log=print) -> IngestReport:
    out = processed_dir()
    (out / "text").mkdir(parents=True, exist_ok=True)
    previous = {}
    if (out / "docs.jsonl").exists():
        for line in (out / "docs.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            previous[rec["doc_id"]] = rec

    docs_out, chunks_out = [], []
    report = IngestReport([], [], [], [], 0)
    for src in load_sources():
        doc_id, path = src["doc_id"], pdf_path(src)
        if src.get("format", "pdf") != "pdf" or not path.exists():
            report.missing.append(doc_id)
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        prev = previous.get(doc_id)
        chunk_file = out / "text" / f"{doc_id}.chunks.jsonl"
        same_input = prev and prev["sha256"] == digest and prev.get("parser_version") == PARSER_VERSION
        if same_input and chunk_file.exists():
            docs_out.append(prev)
            chunks_out.extend(json.loads(x) for x in chunk_file.read_text(encoding="utf-8").splitlines())
            report.reused.append(doc_id)
            continue
        parsed = parse_pdf(path, doc_id, src["title"])
        chunks: list[DocChunk] = chunk_document(parsed)
        (out / "text" / f"{doc_id}.txt").write_text(parsed.text, encoding="utf-8")
        chunk_file.write_text(
            "\n".join(json.dumps(c.as_dict(), ensure_ascii=False) for c in chunks) + "\n", encoding="utf-8"
        )
        n_pages = len(parsed.page_starts)
        rec = {
            "doc_id": doc_id,
            "title": src["title"],
            "publisher": src["publisher"],
            "licence": src["licence"],
            "commit_raw": src["commit_raw"],
            "url": src.get("url"),
            "sha256": digest,
            "parser_version": PARSER_VERSION,
            "pages": n_pages,
            "chars": len(parsed.text),
            "page_starts": parsed.page_starts,
            "n_chunks": len(chunks),
        }
        docs_out.append(rec)
        chunks_out.extend(c.as_dict() for c in chunks)
        report.parsed.append(doc_id)
        if n_pages and len(parsed.text) / n_pages < MIN_CHARS_PER_PAGE:
            report.low_text.append(doc_id)
        log(f"[{doc_id}] {n_pages} pages, {len(chunks)} chunks")

    (out / "docs.jsonl").write_text("\n".join(json.dumps(d) for d in docs_out) + "\n", encoding="utf-8")
    (out / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks_out) + "\n", encoding="utf-8"
    )
    report.n_chunks = len(chunks_out)
    return report


def load_chunks() -> list[DocChunk]:
    path = processed_dir() / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: uv run hyde-rag ingest-food")
    return [DocChunk(**json.loads(x)) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_docs() -> dict[str, dict]:
    path = processed_dir() / "docs.jsonl"
    return {d["doc_id"]: d for d in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if d}
