"""Retrieval metrics with trec_eval semantics (pytrec-eval-terrier), as used by Pyserini.

Per-query values are returned so they can feed the paired bootstrap (eval/stats.py).
"""

from __future__ import annotations

import pytrec_eval

Run = dict[str, dict[str, float]]
Qrels = dict[str, dict[str, int]]

# Our metric name -> trec_eval measure.
_TREC_MEASURES = {"ndcg@10": "ndcg_cut_10", "recall@10": "recall_10", "recall@100": "recall_100"}


def remove_self_hits(run: Run) -> Run:
    """Drop documents whose id equals the query id (ArguAna; Pyserini --remove-query)."""
    return {qid: {d: s for d, s in docs.items() if d != qid} for qid, docs in run.items()}


def truncate(run: Run, k: int) -> Run:
    return {qid: dict(sorted(docs.items(), key=lambda kv: (-kv[1], kv[0]))[:k]) for qid, docs in run.items()}


def capped_recall(qrels: Qrels, run: Run, k: int) -> dict[str, float]:
    """BEIR's R_cap@k = |relevant in top-k| / min(k, |relevant|) (Thakur et al., 2021)."""
    out = {}
    top = truncate(run, k)
    for qid, rels in qrels.items():
        relevant = {d for d, r in rels.items() if r > 0}
        if not relevant:
            continue
        hits = len(relevant & set(top.get(qid, {})))
        out[qid] = hits / min(k, len(relevant))
    return out


def evaluate(
    qrels: Qrels, run: Run, metrics: tuple[str, ...] = ("ndcg@10", "recall@100")
) -> dict[str, dict[str, float]]:
    """Return {metric: {query_id: value}} over every query that has at least one relevant doc.

    Queries missing from the run score 0, as in trec_eval with -c.
    """
    judged = {q: r for q, r in qrels.items() if any(v > 0 for v in r.values())}
    full_run = {q: run.get(q, {}) or {"__none__": 0.0} for q in judged}
    out: dict[str, dict[str, float]] = {}

    trec = {m: _TREC_MEASURES[m] for m in metrics if m in _TREC_MEASURES}
    if trec:
        evaluator = pytrec_eval.RelevanceEvaluator(judged, set(trec.values()))
        per_q = evaluator.evaluate(full_run)
        for name, measure in trec.items():
            out[name] = {q: per_q[q][measure] for q in judged}
    if "mrr@10" in metrics:
        evaluator = pytrec_eval.RelevanceEvaluator(judged, {"recip_rank"})
        per_q = evaluator.evaluate(truncate(full_run, 10))
        out["mrr@10"] = {q: per_q[q]["recip_rank"] for q in judged}
    for m in metrics:
        if m.startswith("rcap@"):
            values = capped_recall(judged, full_run, int(m.split("@")[1]))
            out[m] = {q: values.get(q, 0.0) for q in judged}
    return out


def mean(per_query: dict[str, float]) -> float:
    return sum(per_query.values()) / len(per_query) if per_query else 0.0
