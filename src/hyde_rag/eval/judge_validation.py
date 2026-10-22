"""Validate the LLM judge against human labels before trusting its numbers (D-EVAL-5).

export(): sample claims the judge scored and write a blind labelling sheet. The sheet shows
          each claim, its sentence and the cited passages, but not the judge's verdict.
          Sampling is balanced between the judge's two verdicts, so both kinds of error are
          measured. The sheet holds corpus text, so it goes to data/judge_validation/
          (gitignored); judge verdicts go to a separate key file.
agreement(): compare human labels with the judge: raw agreement, Cohen's kappa, confusion
          matrix -> results/<experiment>/judge_validation.json (committed; no corpus text).
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from hyde_rag import paths
from hyde_rag.ingest.beir import load_beir

LABELS = ("supported", "not_supported")
COLUMNS = ["item_id", "claim", "sentence", "cited_passages", "human_label", "notes"]


def sheet_dir() -> Path:
    return paths.home() / "data" / "judge_validation"


def export(
    experiment: str = "scifact-generation", dataset: str = "scifact", n: int = 50, seed: int = 13
) -> Path:
    items_path = paths.results_dir() / experiment / "items.jsonl"
    items = [json.loads(x) for x in items_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    corpus = load_beir(dataset).corpus
    pool: dict[str, list[dict]] = {lab: [] for lab in LABELS}
    for it in items:
        for k, cl in enumerate(it.get("judge_claims", [])):
            idx = int(cl.get("sentence", -1))
            if cl.get("verdict") not in LABELS or not 0 <= idx < len(it["sentences"]):
                continue
            sent = it["sentences"][idx]
            pool[cl["verdict"]].append(
                {
                    "item_id": f"{it['system']}|{it['qid']}|{idx}|{k}",
                    "claim": cl.get("claim", ""),
                    "sentence": sent["text"],
                    "cited": sent["cited_ids"],
                    "judge": cl["verdict"],
                }
            )
    rng = random.Random(seed)
    half = n // 2
    take_not = min(half, len(pool["not_supported"]))
    sample = rng.sample(pool["not_supported"], take_not)
    sample += rng.sample(pool["supported"], min(n - take_not, len(pool["supported"])))
    rng.shuffle(sample)

    out = sheet_dir()
    out.mkdir(parents=True, exist_ok=True)
    sheet = out / f"{experiment}_judge_sheet.csv"
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for s in sample:
            passages = []
            for cid in s["cited"]:
                doc = corpus.get(cid.split(":")[0])
                if doc:
                    passages.append(f"[{cid}] {doc.title}. {doc.text}")
            w.writerow(
                {
                    "item_id": s["item_id"],
                    "claim": s["claim"],
                    "sentence": s["sentence"],
                    "cited_passages": "\n\n".join(passages),
                    "human_label": "",
                    "notes": "",
                }
            )
    key = {s["item_id"]: s["judge"] for s in sample}
    (out / f"{experiment}_judge_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    return sheet


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == lab for a, _ in pairs) / n) * (sum(b == lab for _, b in pairs) / n) for lab in LABELS)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(experiment: str = "scifact-generation") -> dict:
    out = sheet_dir()
    key = json.loads((out / f"{experiment}_judge_key.json").read_text(encoding="utf-8"))
    pairs, skipped = [], 0
    with (out / f"{experiment}_judge_sheet.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            human = row["human_label"].strip().lower().replace(" ", "_").replace("-", "_")
            human = {"yes": "supported", "no": "not_supported", "y": "supported", "n": "not_supported"}.get(
                human, human
            )
            if human not in LABELS:
                skipped += 1
                continue
            pairs.append((human, key[row["item_id"]]))
    confusion = {
        f"human_{h}/judge_{j}": sum(1 for a, b in pairs if a == h and b == j) for h in LABELS for j in LABELS
    }
    result = {
        "n_labelled": len(pairs),
        "unlabelled": skipped,
        "agreement": sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
        "cohen_kappa": cohen_kappa(pairs),
        "confusion": confusion,
        # Share of claims a person calls supported, given the judge's verdict. The sample is balanced
        # between verdicts, so these rates (not raw agreement) carry over to the full population.
        "human_supported_rate_given_judge": {
            j: (
                sum(1 for a, b in pairs if b == j and a == "supported") / n
                if (n := sum(1 for _, b in pairs if b == j))
                else None
            )
            for j in LABELS
        },
        "note": "Sample balanced between judge verdicts (not population-weighted).",
    }
    dest = paths.results_dir() / experiment / "judge_validation.json"
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def corrected_faithfulness(judge_faithfulness: float, validation: dict) -> float | None:
    """Judge faithfulness re-weighted by how often people agree with each judge verdict:
    P(supported) = P(J=sup)·P(H=sup|J=sup) + P(J=not)·P(H=sup|J=not)."""
    rates = validation.get("human_supported_rate_given_judge") or {}
    a, b = rates.get("supported"), rates.get("not_supported")
    if a is None or b is None:
        return None
    return judge_faithfulness * a + (1.0 - judge_faithfulness) * b
