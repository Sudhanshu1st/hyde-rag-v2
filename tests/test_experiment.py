"""End-to-end experiment run on a tiny synthetic BEIR dataset: no downloads, no API."""

import hashlib
import json

import numpy as np
import pytest
import yaml

from hyde_rag.eval.experiment import Experiment
from hyde_rag.eval.report import render_table
from hyde_rag.llm.cache import JsonlCache
from hyde_rag.llm.client import LLMResponse
from hyde_rag.retrieval.dense import EncoderSpec, encode_corpus, index_path
from hyde_rag.retrieval.hyde import cache_path

DOCS = {
    "d1": ("Vitamin D", "vitamin d supplementation reduces fracture risk in older adults"),
    "d2": ("Stem cells", "hematopoietic stem cells divide asymmetrically"),
    "d3": ("Listeria", "listeria monocytogenes grows at refrigeration temperatures"),
    "d4": ("Cooling", "cooked food must be cooled from 57C to 21C within two hours"),
    "d5": ("Allergens", "undeclared milk allergen triggers a recall"),
}
QUERIES = {
    "q1": "does vitamin d prevent fractures",
    "q2": "how fast must cooked food cool",
    "q3": "listeria cold",
}
QRELS = {"q1": {"d1": 1}, "q2": {"d4": 1}, "q3": {"d3": 1}}


class HashEncoder:
    """Deterministic bag-of-words hashing encoder standing in for a transformer."""

    spec = EncoderSpec("bge-base", "fake/hash", pooling="cls", normalize=True)

    def encode(self, texts, *, is_query, batch_size=16):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class FakeReranker:
    _model = None

    def score(self, pairs, batch_size=32):
        # overlap between query and doc words
        return [len(set(p[2].split()) & set(p[3].lower().split())) + 0.0 for p in pairs]


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDE_RAG_HOME", str(tmp_path))
    ds = tmp_path / "data" / "beir" / "tiny"
    (ds / "qrels").mkdir(parents=True)
    with (ds / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for did, (title, text) in DOCS.items():
            f.write(json.dumps({"_id": did, "title": title, "text": text}) + "\n")
    with (ds / "queries.jsonl").open("w", encoding="utf-8") as f:
        for qid, text in QUERIES.items():
            f.write(json.dumps({"_id": qid, "text": text}) + "\n")
    (ds / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n" + "".join(f"{q}\t{d}\t1\n" for q, r in QRELS.items() for d in r),
        encoding="utf-8",
    )
    # Cached LLM outputs, as the free-tier runner would have written them.
    passages = {
        "q1": "vitamin d fracture risk older adults",
        "q2": "cooled within two hours",
        "q3": "listeria grows cold",
    }
    hyde = JsonlCache(cache_path("tiny", "hyde", "paper.scifact", "fake/gen"))
    mq = JsonlCache(cache_path("tiny", "multi_query", "rewrite_k4", "fake/gen"))
    for q, p in passages.items():
        for s in range(2):
            hyde.put(f"{q}{s}", q, s, LLMResponse(p, "stop", "fake/gen", 40, 60, 0.5 + s))
        mq.put(
            f"mq{q}", q, 0, LLMResponse(f"1. {p}\n2. {QUERIES[q]} evidence", "stop", "fake/gen", 30, 20, 0.3)
        )
    return tmp_path


def test_full_experiment_pipeline(home, monkeypatch):
    enc = HashEncoder()
    monkeypatch.setattr(Experiment, "encoder", lambda self, name: enc)
    monkeypatch.setattr(Experiment, "reranker", lambda self: FakeReranker())
    monkeypatch.setattr(Experiment, "probe_latency", lambda self, spec: 1.0)
    ids = list(DOCS)
    encode_corpus(
        enc, ids, [f"{DOCS[d][0]} {DOCS[d][1]}" for d in ids], index_path("tiny", "bge-base"), progress=False
    )

    cfg = {
        "name": "tiny",
        "dataset": "tiny",
        "depth": 5,
        "baselines": {"bge-base": "dense-bge"},
        "systems": [
            {"name": "bm25", "type": "bm25", "baseline": "dense-bge"},
            {"name": "dense-bge", "type": "dense", "encoder": "bge-base"},
            {
                "name": "hyde-bge-n2",
                "type": "hyde",
                "encoder": "bge-base",
                "generator": "fake/gen",
                "instruction": "paper.scifact",
                "n": 2,
            },
            {
                "name": "hyde-bge-n9",
                "type": "hyde",
                "encoder": "bge-base",
                "generator": "fake/gen",
                "instruction": "paper.scifact",
                "n": 9,
            },  # not enough cached samples -> skipped
            {
                "name": "hybrid-hyde",
                "type": "rrf",
                "inputs": ["bm25", "hyde-bge-n2"],
                "baseline": "dense-bge",
            },
            {
                "name": "q2d",
                "type": "bm25_q2d",
                "generator": "fake/gen",
                "instruction": "paper.scifact",
                "n": 2,
            },
            {
                "name": "mq",
                "type": "multi_query",
                "generator": "fake/gen",
                "legs": ["bm25", "dense:bge-base"],
            },
            {"name": "rerank-hybrid-hyde", "type": "rerank", "input": "hybrid-hyde", "depth": 5},
        ],
    }
    cfg_path = home / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    exp = Experiment(cfg_path)
    logs = []
    exp.run_all(log=logs.append)
    assert "hyde-bge-n9" not in exp.results
    assert any("hyde-bge-n9: skipped" in line for line in logs)
    assert set(exp.results) == {
        "bm25",
        "dense-bge",
        "hyde-bge-n2",
        "hybrid-hyde",
        "q2d",
        "mq",
        "rerank-hybrid-hyde",
    }

    summary = exp.report()
    hyde = summary["systems"]["hyde-bge-n2"]
    assert hyde["llm_tokens_per_query"] == pytest.approx(200)  # 2 samples x 100 tokens
    assert hyde["llm_latency_s"] == pytest.approx(1.5)  # slowest of 2 parallel calls
    assert hyde["metrics"]["ndcg@10"]["baseline"] == "dense-bge"
    assert summary["systems"]["bm25"]["metrics"]["ndcg@10"]["mean"] == pytest.approx(1.0)
    assert (exp.out_dir / "runs" / "hybrid-hyde.json.gz").exists()
    table = render_table(exp.out_dir / "metrics.json")
    assert "| hyde-bge-n2 |" in table and "baseline" in table
