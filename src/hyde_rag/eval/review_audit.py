"""Human audit of an LLM review of the food question set.

When the review decisions in review.csv come from an LLM (their notes start with LLM_TAG), a
person checks a random sample of them: kept rows (is the question clear and the answer right?)
and rejected rows (was rejecting it right?). The agreement rate is reported with the results,
so the eval set is described as "LLM-reviewed, human-audited", never as human-reviewed.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from hyde_rag.eval.question_gen import EVALSET_DIR

LLM_TAG = "[llm-review]"
AUDIT_COLUMNS = [
    "id",
    "type",
    "decision",
    "question",
    "reference_answer",
    "evidence_doc",
    "evidence_quote",
    "reviewer_notes",
    "human_verdict",  # agree / disagree
    "human_notes",
]
VERDICTS = {"agree", "disagree"}


def _final(row: dict) -> dict:
    """The row as it would enter the eval set (edits applied)."""
    return {
        "id": row["id"],
        "type": row["type"],
        "decision": row["decision"].strip().lower(),
        "question": (row.get("edited_question") or "").strip() or row["question"],
        "reference_answer": (row.get("edited_answer") or "").strip() or row["reference_answer"],
        "evidence_doc": row["evidence_doc"],
        "evidence_quote": row["evidence_quote"],
        "reviewer_notes": row["notes"].removeprefix(LLM_TAG).strip(),
    }


def export(n_kept: int = 30, n_rejected: int = 10, seed: int = 7, review_csv: Path | None = None) -> Path:
    """Write audit.csv with a random sample of LLM decisions; verdicts already filled are kept."""
    review_csv = review_csv or EVALSET_DIR / "review.csv"
    with review_csv.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["type"] != "human" and r["notes"].startswith(LLM_TAG)]
    kept = sorted(
        (r for r in rows if r["decision"].strip().lower() in ("accept", "edit")), key=lambda r: r["id"]
    )
    rejected = sorted((r for r in rows if r["decision"].strip().lower() == "reject"), key=lambda r: r["id"])
    rng = random.Random(seed)
    sample = rng.sample(kept, min(n_kept, len(kept))) + rng.sample(rejected, min(n_rejected, len(rejected)))
    rng.shuffle(sample)

    out = review_csv.parent / "audit.csv"
    previous: dict[str, dict] = {}
    if out.exists():
        with out.open(encoding="utf-8-sig", newline="") as f:
            previous = {r["id"]: r for r in csv.DictReader(f)}
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        w.writeheader()
        for r in sample:
            item = _final(r)
            old = previous.get(item["id"], {})
            item["human_verdict"] = old.get("human_verdict", "")
            item["human_notes"] = old.get("human_notes", "")
            w.writerow(item)
    return out


def agreement(audit_csv: Path | None = None) -> dict:
    """Share of audited LLM decisions the human agrees with, overall and for kept / rejected rows."""
    audit_csv = audit_csv or EVALSET_DIR / "audit.csv"
    with audit_csv.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["human_verdict"].strip().lower() in VERDICTS]

    def rate(subset: list[dict]) -> float | None:
        if not subset:
            return None
        return sum(r["human_verdict"].strip().lower() == "agree" for r in subset) / len(subset)

    kept = [r for r in rows if r["decision"] in ("accept", "edit")]
    rejected = [r for r in rows if r["decision"] == "reject"]
    result = {
        "n_labelled": len(rows),
        "agreement": rate(rows),
        "agreement_kept": rate(kept),
        "agreement_rejected": rate(rejected),
        "n_kept": len(kept),
        "n_rejected": len(rejected),
        "disagreements": sorted(r["id"] for r in rows if r["human_verdict"].strip().lower() == "disagree"),
    }
    (audit_csv.parent / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
