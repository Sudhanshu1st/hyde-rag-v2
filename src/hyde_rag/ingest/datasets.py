"""One loader for every retrieval dataset: BEIR (scifact, trec-covid, ...) and the food set.

The food set becomes a BEIR-shaped dataset:
  corpus  = chunks (doc title + section path as the title, chunk text as the text)
  queries = eval-set questions of a split (dev / test)
  qrels   = chunks overlapping a question's gold evidence spans (span-based, so evaluation
            does not depend on chunking); unanswerable questions have no qrels and are
            ignored by the retrieval metrics (they are scored by the abstention metrics).
"""

from __future__ import annotations

import json

from hyde_rag import paths
from hyde_rag.ingest.beir import BeirDataset, Doc, load_beir
from hyde_rag.ingest.chunking import overlaps

FOOD = "food"


def collection(name: str) -> str:
    """Cache collection for a dataset: cache/<collection>/<dataset>/..."""
    return "food" if name == FOOD else "beir"


def load_food_eval(split: str) -> BeirDataset:
    from hyde_rag.ingest.food import load_chunks, load_docs

    chunks = load_chunks()
    titles = {d: v["title"] for d, v in load_docs().items()}
    corpus = {
        c.chunk_id: Doc(title=f"{titles[c.doc_id]} | {c.section}".rstrip(" |"), text=c.text) for c in chunks
    }
    path = paths.REPO_ROOT / "eval_sets" / "food_quality" / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Review eval_sets/food_quality/review.csv, then run build-evalset."
        )
    queries, qrels = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        queries[item["id"]] = item["question"]
        rel = {}
        for ev in item.get("gold_evidence", []):
            for c in chunks:
                if overlaps(c, ev["doc_id"], ev["char_start"], ev["char_end"]):
                    rel[c.chunk_id] = 1
        qrels[item["id"]] = rel
    return BeirDataset(name=FOOD, split=split, corpus=corpus, queries=queries, qrels=qrels)


def load_dataset(name: str, split: str = "test") -> BeirDataset:
    return load_food_eval(split) if name == FOOD else load_beir(name, split=split)
