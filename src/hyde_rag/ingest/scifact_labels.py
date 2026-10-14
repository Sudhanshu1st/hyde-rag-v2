"""SciFact claim labels from the original release (scripts/download_scifact_labels.py).

BEIR's SciFact test split is the original dev set, and BEIR's train split the original train
set; claim ids are shared (D-EVAL-6).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

from hyde_rag import paths

GOLD_VERDICT = {"SUPPORT": "SUPPORTED", "CONTRADICT": "REFUTED", "NONE": "NOT FOUND"}


def load_labels(split: str = "test") -> dict[str, str]:
    """claim id -> SUPPORT | CONTRADICT | NONE (no evidence). Mixed-label claims are skipped."""
    fname = {"test": "claims_dev.jsonl", "train": "claims_train.jsonl"}[split]
    path = paths.home() / "data" / "scifact_labels" / fname
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python scripts/download_scifact_labels.py")
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        claim = json.loads(line)
        found = {e["label"] for ev in claim.get("evidence", {}).values() for e in ev}
        if len(found) > 1:
            continue
        labels[str(claim["id"])] = found.pop() if found else "NONE"
    return labels


def stratified_sample(labels: dict[str, str], n: int, seed: int) -> list[str]:
    """Sample n claim ids keeping the label mix (largest-remainder allocation)."""
    by_label: dict[str, list[str]] = defaultdict(list)
    for qid in sorted(labels):
        by_label[labels[qid]].append(qid)
    total = len(labels)
    quotas = {lab: n * len(ids) / total for lab, ids in by_label.items()}
    alloc = {lab: int(q) for lab, q in quotas.items()}
    for lab in sorted(quotas, key=lambda k: quotas[k] - alloc[k], reverse=True)[: n - sum(alloc.values())]:
        alloc[lab] += 1
    rng = random.Random(seed)
    picked = [q for lab in sorted(by_label) for q in rng.sample(by_label[lab], alloc[lab])]
    return sorted(picked)
