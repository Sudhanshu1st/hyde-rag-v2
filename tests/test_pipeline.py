"""Pipeline + API end to end with a fake encoder and a scripted LLM (no models, no network)."""

import hashlib
import json

import numpy as np
import pytest

from hyde_rag import pipeline as pl
from hyde_rag.generation.answer import ABSTAIN_TEXT
from hyde_rag.llm.client import LLMResponse
from hyde_rag.retrieval.dense import EncoderSpec

CHUNKS = [
    (
        "fc",
        87,
        0,
        "3-501.14 Cooling",
        "Cooked food shall be cooled from 57C to 21C within 2 hours and to 5C within 6 hours.",
    ),
    (
        "fc",
        88,
        0,
        "3-501.15 Cooling Methods",
        "Cooling methods include ice baths, shallow pans and blast chillers.",
    ),
    (
        "lm",
        12,
        0,
        "Environmental monitoring",
        "Swab food contact surfaces for Listeria at least weekly in RTE areas.",
    ),
]


class HashEncoder:
    spec = EncoderSpec("bge-base", "fake/hash", pooling="cls", normalize=True)

    def __init__(self, *a, **k):
        pass

    def encode(self, texts, *, is_query, batch_size=16):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace(".", " ").split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class FakeLLM:
    base_url = "https://fake.local/v1"

    def __init__(self):
        self.calls = []

    def complete(self, request):
        self.calls.append(request)
        user = request.messages[-1][1]
        if "Passage:" in user and "Context:" not in user:  # HyDE instruction
            text = "**Cooling** cooked food must be cooled from 57C to 21C within 2 hours."
        elif "pizza" in user.lower():
            text = ABSTAIN_TEXT
        else:
            text = "Cool from 57C to 21C within 2 hours [fc:87:0]. Use ice baths or shallow pans [fc:88:0]."
        return LLMResponse(text, "stop", request.model, 100, 50, 0.2)


@pytest.fixture()
def pipe(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDE_RAG_HOME", str(tmp_path))
    proc = tmp_path / "data" / "food" / "processed"
    proc.mkdir(parents=True)
    rows = [
        {
            "chunk_id": f"{d}:{p}:{c}",
            "doc_id": d,
            "page": p,
            "chunk": c,
            "section": s,
            "text": x,
            "char_start": 0,
            "char_end": len(x),
        }
        for d, p, c, s, x in CHUNKS
    ]
    (proc / "chunks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    docs = [{"doc_id": "fc", "title": "Food Code 2026"}, {"doc_id": "lm", "title": "Listeria guidance"}]
    (proc / "docs.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8")
    monkeypatch.setattr(pl, "Encoder", HashEncoder)
    pl.build_food_index("bge-base")
    llm = FakeLLM()
    return pl.RAGPipeline(pl.PipelineConfig(mode="hyde", n_passages=2, top_k=2), llm=llm), llm


def test_pipeline_returns_answer_citations_and_hyde_trace(pipe):
    p, llm = pipe
    res = p.query("How fast must cooked food be cooled?")
    assert not res.abstained and res.citations[0]["cited_ids"] == ["fc:87:0"]
    assert res.invalid_citations == [] and res.uncited_sentences == []
    assert len(res.hypothetical_documents) == 2 and "**" not in res.hypothetical_documents[0]  # cleaned
    assert res.retrieved[0]["chunk_id"].startswith("fc:")
    assert {"hyde_llm", "dense", "answer_llm", "total"} <= set(res.timings_ms)
    assert res.llm_tokens == 3 * 150  # 2 HyDE samples + 1 answer
    assert len(llm.calls) == 3


def test_pipeline_abstains_and_flags_bad_citations(pipe):
    p, llm = pipe
    res = p.query("Best pizza topping?")
    assert res.abstained and not res.gated
    p.cfg.mode = "dense"
    llm.complete = lambda req: LLMResponse("Cool quickly [zz:1:1]. No source here.", "stop", "m", 10, 10, 0.1)
    bad = p.query("How fast must cooked food be cooled?")
    assert bad.invalid_citations == ["zz:1:1"] and bad.uncited_sentences == [1]
    assert bad.hypothetical_documents == []


def test_api_query_endpoint(pipe, monkeypatch):
    from fastapi.testclient import TestClient

    from hyde_rag.api import app as api

    p, _ = pipe
    monkeypatch.setattr(api, "_pipeline", lambda mode, n: p)
    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok"}
    r = client.post("/query", json={"question": "How fast must cooked food be cooled?", "mode": "hyde"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"][0]["cited_ids"] == ["fc:87:0"] and body["hypothetical_documents"]
    assert client.post("/query", json={"question": "x?", "mode": "bogus"}).status_code == 422


def test_pipeline_reranks_the_top_n_with_the_cross_encoder(pipe):
    p, _ = pipe

    class FakeCE:
        def predict(self, pairs):  # prefers the Listeria guidance, whatever the retriever said
            return [5.0 if t.startswith("Listeria guidance") else 0.0 for _, t in pairs]

    p._gate_model = FakeCE()
    p.cfg.mode, p.cfg.rerank_depth = "dense", 3
    res = p.query("How fast must cooked food be cooled?")
    assert res.retrieved[0]["chunk_id"].startswith("lm:") and "rerank" in res.timings_ms
