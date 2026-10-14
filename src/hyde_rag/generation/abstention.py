"""Calibrate the retrieval-gate threshold tau on the dev split only."""

from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class GateCalibration:
    tau: float | None  # None = gate disabled (no threshold beat doing nothing)
    f1: float
    precision: float
    recall: float
    false_refusal: float


def calibrate_tau(
    answerable_scores: list[float],
    unanswerable_scores: list[float],
    *,
    max_false_refusal: float = 0.05,
) -> GateCalibration:
    """Pick tau maximising F1 of "should abstain" detection, subject to a false-refusal cap.

    A query abstains when its top reranker score is < tau. Candidate thresholds are midpoints
    between consecutive distinct observed scores, so tau sits between the two groups instead of
    on the edge of the dev data (which would refuse slightly lower-scoring answerable queries at
    test time). Ties prefer the lower tau (fewer refusals).
    """
    best = GateCalibration(None, 0.0, 0.0, 0.0, 0.0)
    n_ans, n_un = len(answerable_scores), len(unanswerable_scores)
    if not n_ans or not n_un:
        return best
    values = sorted(set(answerable_scores) | set(unanswerable_scores))
    candidates = [(a + b) / 2 for a, b in itertools.pairwise(values)]
    for tau in candidates:
        tp = sum(s < tau for s in unanswerable_scores)
        fp = sum(s < tau for s in answerable_scores)
        if tp == 0:
            continue
        false_refusal = fp / n_ans
        if false_refusal > max_false_refusal:
            continue
        precision, recall = tp / (tp + fp), tp / n_un
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best.f1 + 1e-12:
            best = GateCalibration(tau, f1, precision, recall, false_refusal)
    return best
