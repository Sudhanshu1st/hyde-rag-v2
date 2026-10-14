"""Load BEIR datasets downloaded by scripts/download_beir.py."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from hyde_rag import paths


@dataclass(frozen=True)
class Doc:
    title: str
    text: str


@dataclass(frozen=True)
class BeirDataset:
    name: str
    split: str
    corpus: dict[str, Doc]
    queries: dict[str, str]  # only queries that have qrels in this split
    qrels: dict[str, dict[str, int]]


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: query-id, corpus-id, score
        for qid, did, score in reader:
            # TREC-COVID has two -1 judgements; treat them as non-relevant (D-EVAL-2).
            qrels.setdefault(qid, {})[did] = max(0, int(score))
    return qrels


def load_beir(name: str, split: str = "test", root: Path | None = None) -> BeirDataset:
    ds_dir = (root or paths.beir_dir()) / name
    if not (ds_dir / "corpus.jsonl").exists():
        raise FileNotFoundError(f"{ds_dir} not found. Run: python scripts/download_beir.py {name}")
    qrels = load_qrels(ds_dir / "qrels" / f"{split}.tsv")

    corpus: dict[str, Doc] = {}
    with (ds_dir / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            corpus[str(obj["_id"])] = Doc(title=obj.get("title") or "", text=obj.get("text") or "")

    queries: dict[str, str] = {}
    with (ds_dir / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj["_id"])
            if qid in qrels:
                queries[qid] = obj["text"]
    return BeirDataset(name=name, split=split, corpus=corpus, queries=queries, qrels=qrels)
