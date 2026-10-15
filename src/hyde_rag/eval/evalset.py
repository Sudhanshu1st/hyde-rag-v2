"""Build the food-quality eval set from the human-reviewed CSV.

Only reviewed items enter the eval set. For each row of eval_sets/food_quality/review.csv:
  decision = accept     -> keep as generated
  decision = edit       -> use edited_question / edited_answer (and the evidence quote as edited)
  decision = reject/""  -> dropped
  type = human          -> a question you wrote: fill question, answerable, reference_answer,
                           evidence_doc and a verbatim evidence_quote (|| separates several)
Gold evidence is stored as document char spans (doc_id, page, section, char_start, char_end),
so evaluation doesn't depend on chunking. Answerable items whose evidence can't be located are
excluded and listed. Split: 30% dev / 70% test, stratified by type, fixed seed.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from hyde_rag.eval.question_gen import EVALSET_DIR, locate
from hyde_rag.ingest.chunking import DocChunk
from hyde_rag.ingest.food import load_chunks, processed_dir

DEV_SHARE = 0.3
SEED = 13
TRUE = {"true", "yes", "y", "1"}


@dataclass
class EvalItem:
    id: str
    question: str
    type: str
    answerable: bool
    gold_evidence: list[dict]
    reference_answer: str
    source: str  # "generated" | "human"


@dataclass
class BuildReport:
    kept: int = 0
    rejected: int = 0
    unreviewed: int = 0
    excluded: list[str] = field(default_factory=list)  # answerable items without located evidence
    by_split: dict = field(default_factory=dict)


def _locate_in_doc(quote: str, doc_id: str, chunks: list[DocChunk], text: str) -> dict | None:
    for c in chunks:
        if c.doc_id != doc_id:
            continue
        span = locate(quote, c, text)
        if span:
            return {
                "doc_id": doc_id,
                "page": c.page,
                "section": c.section,
                "char_start": span[0],
                "char_end": span[1],
            }
    return None


def build(review_csv: Path | None = None) -> BuildReport:
    review_csv = review_csv or EVALSET_DIR / "review.csv"
    candidates = {}
    cand_path = EVALSET_DIR / "candidates.jsonl"
    if cand_path.exists():
        for line in cand_path.read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            candidates[c["id"]] = c
    chunks = load_chunks()
    texts: dict[str, str] = {}

    def doc_text(doc_id: str) -> str:
        if doc_id not in texts:
            texts[doc_id] = (processed_dir() / "text" / f"{doc_id}.txt").read_text(encoding="utf-8")
        return texts[doc_id]

    items: list[EvalItem] = []
    report = BuildReport()
    with review_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            decision = (row.get("decision") or "").strip().lower()
            human = row["type"] == "human"
            if human and not row["question"].strip():
                continue  # unused blank row
            if decision == "reject":
                report.rejected += 1
                continue
            if not human and decision not in ("accept", "edit"):
                report.unreviewed += 1
                continue
            question = (row.get("edited_question") or "").strip() or row["question"].strip()
            answer = (row.get("edited_answer") or "").strip() or row["reference_answer"].strip()
            answerable = str(row["answerable"]).strip().lower() in TRUE
            evidence: list[dict] = []
            if answerable:
                cand = candidates.get(row["id"])
                quotes = [q.strip() for q in row["evidence_quote"].split("||") if q.strip()]
                original = " || ".join(e["quote"] for e in cand["evidence"]) if cand else None
                if cand and row["evidence_quote"].strip() == (original or "").strip():
                    evidence = [
                        {k: e[k] for k in ("doc_id", "page", "section", "char_start", "char_end")}
                        for e in cand["evidence"]
                        if e["located"]
                    ]
                else:  # edited or human-written evidence: locate it again
                    doc_id = row["evidence_doc"].strip()
                    for q in quotes:
                        hit = _locate_in_doc(q, doc_id, chunks, doc_text(doc_id)) if doc_id else None
                        if hit:
                            evidence.append(hit)
                if not evidence:
                    report.excluded.append(row["id"])
                    continue
            qtype = "human" if human else row["type"]  # human-written items are reported separately
            items.append(
                EvalItem(
                    row["id"],
                    question,
                    qtype,
                    answerable,
                    evidence,
                    answer,
                    "human" if human else "generated",
                )
            )

    # Stratified 30/70 split by type (human-written items form their own stratum).
    rng = random.Random(SEED)
    by_type: dict[str, list[EvalItem]] = {}
    for it in items:
        by_type.setdefault(it.type, []).append(it)
    dev, test = [], []
    for _, group in sorted(by_type.items()):
        group = sorted(group, key=lambda x: x.id)
        rng.shuffle(group)
        n_dev = round(DEV_SHARE * len(group))
        dev += group[:n_dev]
        test += group[n_dev:]
    for name, split in (("dev", dev), ("test", test)):
        path = EVALSET_DIR / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(it.__dict__, ensure_ascii=False) + "\n" for it in split), encoding="utf-8"
        )
        report.by_split[name] = {
            t: sum(1 for x in split if x.type == t) for t in sorted({x.type for x in split})
        }
    report.kept = len(items)
    return report
