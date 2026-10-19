"""FastAPI service: POST /query returns the answer, citations, the HyDE trace and timings.

Run:  uv run uvicorn hyde_rag.api.app:app --port 8000
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hyde_rag.pipeline import MODES, PipelineConfig, RAGPipeline, load_pipeline_config

app = FastAPI(
    title="HyDE RAG v2",
    description="Grounded answers over public food-safety documents, with an inspectable HyDE trace.",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    mode: str | None = Field(
        None, description=f"one of {', '.join(MODES)} (default from configs/pipeline.yaml)"
    )
    n_passages: int | None = Field(None, ge=1, le=8)


class Citation(BaseModel):
    sentence: str
    cited_ids: list[str]


class Retrieved(BaseModel):
    chunk_id: str
    title: str
    section: str
    page: int
    gate_score: float | None
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    gated: bool
    citations: list[Citation]
    invalid_citations: list[str]
    uncited_sentences: list[int]
    hypothetical_documents: list[str]
    retrieved: list[Retrieved]
    timings_ms: dict[str, float]
    llm_tokens: int
    config: dict


@lru_cache(maxsize=8)
def _pipeline(mode: str, n_passages: int) -> RAGPipeline:
    base = load_pipeline_config()
    cfg = PipelineConfig(**{**base.__dict__, "mode": mode, "n_passages": n_passages})
    return RAGPipeline(cfg)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    base = load_pipeline_config()
    mode = req.mode or base.mode
    if mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    try:
        pipe = _pipeline(mode, req.n_passages or base.n_passages)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    return QueryResponse(**pipe.query(req.question).to_dict())
