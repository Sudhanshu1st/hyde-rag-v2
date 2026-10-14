import numpy as np
import pytest

from hyde_rag.eval.metrics import capped_recall, evaluate, mean, remove_self_hits
from hyde_rag.eval.stats import bootstrap_ci, compare_paired, holm
from hyde_rag.retrieval.bm25 import BM25Retriever, LuceneLikeAnalyzer, top_k
from hyde_rag.retrieval.fusion import hyde_vector, rrf
from hyde_rag.retrieval.hyde import clean_passage, parse_rewrites


def test_analyzer_mirrors_lucene_english():
    a = LuceneLikeAnalyzer()
    # lowercase, possessive stripped, stopwords dropped, Porter-stemmed; single chars kept
    assert a("The Cell's proteins are binding to a B receptor") == [
        "cell",
        "protein",
        "bind",
        "b",
        "receptor",
    ]


def test_bm25_ranks_the_matching_doc_first():
    r = BM25Retriever()
    r.index(["d1", "d2", "d3"], ["vitamin d and bone fractures", "stem cell division", "bone marrow cells"])
    run = r.search({"q": "fractures of bone"}, k=3)
    assert next(iter(sorted(run["q"], key=run["q"].get, reverse=True))) == "d1"
    assert r.search({"q": "zzzz"})["q"] == {}  # unknown terms only -> empty, no crash


def test_metrics_match_hand_computation():
    qrels = {"q1": {"a": 1, "b": 0}, "q2": {"c": 2, "d": 1}}
    run = {"q1": {"x": 3.0, "a": 2.0}, "q2": {"d": 5.0, "c": 4.0}}
    m = evaluate(qrels, run, ("ndcg@10", "recall@10", "mrr@10"))
    assert m["mrr@10"] == {"q1": 0.5, "q2": 1.0}
    assert m["recall@10"]["q1"] == 1.0
    assert m["ndcg@10"]["q1"] == pytest.approx(1 / np.log2(3))  # one relevant doc at rank 2
    assert mean(m["recall@10"]) == 1.0


def test_queries_missing_from_run_score_zero():
    m = evaluate({"q1": {"a": 1}, "q2": {"b": 1}}, {"q1": {"a": 1.0}}, ("ndcg@10",))
    assert m["ndcg@10"] == {"q1": 1.0, "q2": 0.0}


def test_capped_recall_and_self_hits():
    qrels = {"q": {f"d{i}": 1 for i in range(200)}}
    run = {"q": {f"d{i}": 1.0 / (i + 1) for i in range(100)}}
    assert capped_recall(qrels, run, 100)["q"] == 1.0  # 100 hits / min(100, 200)
    assert remove_self_hits({"q": {"q": 9.0, "d": 1.0}}) == {"q": {"d": 1.0}}


def test_rrf_and_top_k():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert list(fused)[:2] in (["a", "b"], ["b", "a"])
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert top_k(np.array([[0.1, 0.9, 0.5]]), 2).tolist() == [[1, 2]]


def test_hyde_vector_is_eq8_mean_including_query():
    q = np.array([1.0, 0.0])
    docs = np.array([[0.0, 1.0], [0.0, 3.0]])
    assert hyde_vector(q, docs).tolist() == pytest.approx([1 / 3, 4 / 3])


def test_stats():
    base = {f"q{i}": 0.5 for i in range(50)}
    better = {f"q{i}": 0.6 for i in range(50)}
    mean_, lo, hi = bootstrap_ci(better)
    assert (mean_, lo, hi) == pytest.approx((0.6, 0.6, 0.6))
    cmp_ = compare_paired(better, base)
    assert cmp_.delta == pytest.approx(0.1) and cmp_.p_value < 0.001
    same = compare_paired(base, base)
    assert same.p_value == 1.0
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


MARKDOWN_PASSAGE = """## **Abstract**

The claim that `0-D` materials

**induce**  growth."""

NUMBERED_REWRITES = """1. first query
- second

3) third
fourth
fifth"""


def test_clean_passage_and_rewrites():
    assert clean_passage(MARKDOWN_PASSAGE) == "Abstract The claim that 0-D materials induce growth."
    assert parse_rewrites(NUMBERED_REWRITES) == ["first query", "second", "third", "fourth"]
