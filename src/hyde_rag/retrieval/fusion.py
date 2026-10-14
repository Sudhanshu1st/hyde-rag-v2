"""Rank fusion and the HyDE query vector."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Run = dict[str, dict[str, float]]

RRF_K = 60  # Cormack, Clarke & Buettcher (2009)


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K, depth: int = 1000) -> dict[str, float]:
    """Reciprocal Rank Fusion of ranked doc-id lists (best first): sum of 1 / (k + rank)."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking[:depth], start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return dict(sorted(fused.items(), key=lambda kv: (-kv[1], kv[0])))


def rrf_runs(runs: Sequence[Run], k: int = RRF_K, depth: int = 1000) -> Run:
    """Fuse whole runs query by query."""
    qids = set().union(*runs)
    out: Run = {}
    for q in qids:
        rankings = [
            [d for d, _ in sorted(run.get(q, {}).items(), key=lambda kv: (-kv[1], kv[0]))] for run in runs
        ]
        out[q] = rrf(rankings, k=k, depth=depth)
    return out


def hyde_vector(query_vec: np.ndarray, passage_vecs: np.ndarray) -> np.ndarray:
    """Paper Eq. 8: v = (1 / (N + 1)) * (sum_k f(d_k) + f(q)). No normalisation (D-HYDE-1)."""
    if passage_vecs.ndim != 2 or passage_vecs.shape[1] != query_vec.shape[-1]:
        raise ValueError("passage_vecs must be [N, dim] matching the query vector.")
    return np.vstack([query_vec[None, :], passage_vecs]).mean(axis=0)
