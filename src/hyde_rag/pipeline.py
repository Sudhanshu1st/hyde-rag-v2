"""End-to-end RAG pipeline over the food-safety corpus (used by the API and the demo).

question -> [HyDE passages] -> retrieval (dense / HyDE / hybrid) -> abstention gate
         -> grounded answer with inline citations -> citation validation

Everything the user should be able to inspect is returned: the hypothetical documents, the
retrieved chunks, the cited sentences, invalid citations, timings and token usage.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from hyde_rag import paths
from hyde_rag.generation.answer import build_request, gate, gated_answer, parse_answer
from hyde_rag.generation.context import Chunk
from hyde_rag.ingest.chunking import DocChunk
from hyde_rag.ingest.food import load_chunks, load_docs
from hyde_rag.llm.client import LLMClient
from hyde_rag.retrieval.bm25 import BM25Retriever
from hyde_rag.retrieval.dense import ENCODERS, DenseIndex, Encoder, encode_corpus, load_index
from hyde_rag.retrieval.fusion import hyde_vector, rrf
from hyde_rag.retrieval.hyde import clean_passage, hyde_items, load_instruction

FOOD = "food"
MODES = ("dense", "hyde", "hybrid", "hybrid-hyde")


@dataclass
class PipelineConfig:
    encoder: str = "bge-base"
    mode: str = "hyde"
    n_passages: int = 1  # one passage gave nearly all of HyDE's gain on BEIR
    query_weight: float | None = 0.5  # D-HYDE-8; None = paper Eq. 8
    instruction: str = "domain.food_quality_v1"
    top_k: int = 5
    depth: int = 100
    tau: float | None = None  # abstention gate (cross-encoder logit), calibrated on the dev split
    hyde_model: str | None = None  # default: configs/llm.yaml roles.hyde_main
    answer_model: str | None = None  # default: roles.answerer


@dataclass
class PipelineResult:
    question: str
    answer: str
    abstained: bool
    gated: bool
    citations: list[dict] = field(default_factory=list)  # [{sentence, cited_ids}]
    invalid_citations: list[str] = field(default_factory=list)
    uncited_sentences: list[int] = field(default_factory=list)
    hypothetical_documents: list[str] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    llm_tokens: int = 0
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_food_index(encoder: str = "bge-base", threads: int | None = 4) -> Path:
    """Encode every food chunk (title and section header + text) into data/indexes/food/<encoder>/."""
    chunks = load_chunks()
    titles = {d: v["title"] for d, v in load_docs().items()}
    out = paths.index_dir() / FOOD / encoder
    encode_corpus(
        Encoder(ENCODERS[encoder], threads=threads),
        [c.chunk_id for c in chunks],
        [c.index_text(titles[c.doc_id]) for c in chunks],
        out,
    )
    return out


class RAGPipeline:
    def __init__(
        self, cfg: PipelineConfig | None = None, llm: LLMClient | None = None, threads: int | None = 4
    ):
        self.cfg = cfg or PipelineConfig()
        if self.cfg.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        roles = yaml.safe_load((paths.CONFIG_DIR / "llm.yaml").read_text(encoding="utf-8"))
        self.llm_settings = roles.get("generation", {})
        self.hyde_model = self.cfg.hyde_model or roles["roles"]["hyde_main"]
        self.answer_model = self.cfg.answer_model or roles["roles"]["answerer"]
        self._llm = llm
        self.chunks: list[DocChunk] = load_chunks()
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.titles = {d: v["title"] for d, v in load_docs().items()}
        self.encoder = Encoder(ENCODERS[self.cfg.encoder], threads=threads)
        index = load_index(FOOD, self.cfg.encoder)
        if index is None:
            raise FileNotFoundError(
                f"No food index for {self.cfg.encoder}. "
                f"Run: uv run hyde-rag index-food --encoders {self.cfg.encoder}"
            )
        self.index: DenseIndex = index
        self.bm25 = BM25Retriever()
        self.bm25.index(
            [c.chunk_id for c in self.chunks], [c.index_text(self.titles[c.doc_id]) for c in self.chunks]
        )
        self._gate_model = None

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            from hyde_rag.llm.client import OpenAICompatClient

            self._llm = OpenAICompatClient()
        return self._llm

    def _reasoning(self, model: str) -> str | None:
        return self.llm_settings.get(model, {}).get("reasoning_effort")

    def _hyde(self, question: str) -> tuple[list[str], int]:
        items = hyde_items(
            {"q": question}, load_instruction(self.cfg.instruction), self.hyde_model, self.cfg.n_passages
        )
        with ThreadPoolExecutor(max_workers=len(items)) as pool:  # N samples in parallel
            responses = list(pool.map(lambda it: self.llm.complete(it.request), items))
        return [clean_passage(r.text) for r in responses], sum(r.total_tokens for r in responses)

    def _gate_scores(self, question: str, chunks: list[Chunk]) -> list[float]:
        if self._gate_model is None:
            from sentence_transformers import CrossEncoder

            from hyde_rag.retrieval.rerank import DEFAULT_RERANKER

            self._gate_model = CrossEncoder(DEFAULT_RERANKER, device="cpu")
        return [float(s) for s in self._gate_model.predict([(question, c.text) for c in chunks[:3]])]

    def query(self, question: str) -> PipelineResult:
        cfg, t = self.cfg, {}
        res = PipelineResult(question=question, answer="", abstained=False, gated=False, config=asdict(cfg))
        start = time.perf_counter()

        passages: list[str] = []
        if cfg.mode in ("hyde", "hybrid-hyde"):
            s = time.perf_counter()
            passages, tokens = self._hyde(question)
            res.llm_tokens += tokens
            t["hyde_llm"] = (time.perf_counter() - s) * 1000
        res.hypothetical_documents = passages

        s = time.perf_counter()
        qv = self.encoder.encode([question], is_query=True)[0]
        if passages:
            qv = hyde_vector(qv, self.encoder.encode(passages, is_query=False), cfg.query_weight)
        dense = [d for d, _ in self.index.search(qv[None, :], k=cfg.depth)[0]]
        t["dense"] = (time.perf_counter() - s) * 1000
        if cfg.mode.startswith("hybrid"):
            s = time.perf_counter()
            sparse = self.bm25.search({"q": question}, k=cfg.depth)["q"]
            ranked = list(rrf([dense, sorted(sparse, key=lambda d: -sparse[d])], depth=cfg.depth))
            t["bm25_rrf"] = (time.perf_counter() - s) * 1000
        else:
            ranked = dense

        top = [self.by_id[cid] for cid in ranked[: cfg.top_k]]
        chunks = [
            Chunk(c.doc_id, c.page, c.chunk, c.text, title=self.titles[c.doc_id], section=c.section)
            for c in top
        ]
        if cfg.tau is not None:
            s = time.perf_counter()
            scores = self._gate_scores(question, chunks)
            chunks = [
                Chunk(
                    c.doc_id,
                    c.page,
                    c.chunk,
                    c.text,
                    c.title,
                    c.section,
                    score=(scores[i] if i < len(scores) else None),
                )
                for i, c in enumerate(chunks)
            ]
            t["gate"] = (time.perf_counter() - s) * 1000
        res.retrieved = [
            {
                "chunk_id": c.chunk_id,
                "title": c.title,
                "section": c.section,
                "page": c.page,
                "gate_score": c.score,
                "text": c.text,
            }
            for c in chunks
        ]

        s = time.perf_counter()
        if gate(chunks, cfg.tau):
            answer = gated_answer(chunks)
        else:
            req = build_request(
                question, chunks, self.answer_model, reasoning_effort=self._reasoning(self.answer_model)
            )
            r = self.llm.complete(req)
            res.llm_tokens += r.total_tokens
            answer = parse_answer(r.text, chunks)
        t["answer_llm"] = (time.perf_counter() - s) * 1000
        res.answer, res.abstained, res.gated = answer.text, answer.abstained, answer.gated
        res.citations = [{"sentence": x.text, "cited_ids": x.cited_ids} for x in answer.citations.sentences]
        res.invalid_citations = answer.citations.unknown_ids
        res.uncited_sentences = answer.citations.uncited
        t["total"] = (time.perf_counter() - start) * 1000
        res.timings_ms = {k: round(v, 1) for k, v in t.items()}
        return res


def load_pipeline_config(path: Path | None = None) -> PipelineConfig:
    path = path or paths.CONFIG_DIR / "pipeline.yaml"
    if not path.exists():
        return PipelineConfig()
    return PipelineConfig(**(yaml.safe_load(path.read_text(encoding="utf-8")) or {}))


def result_json(res: PipelineResult) -> str:
    return json.dumps(
        res.to_dict(),
        ensure_ascii=False,
        indent=2,
        default=lambda o: float(o) if isinstance(o, np.floating) else str(o),
    )
