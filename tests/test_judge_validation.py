import csv
import json

import pytest

from hyde_rag.eval import judge_validation as jv


def test_cohen_kappa():
    assert jv.cohen_kappa([("supported", "supported"), ("not_supported", "not_supported")]) == 1.0
    assert jv.cohen_kappa([("supported", "not_supported"), ("not_supported", "supported")]) == -1.0
    pairs = [("supported", "supported")] * 8 + [("supported", "not_supported")] * 2
    assert jv.cohen_kappa(pairs) == pytest.approx(0.0)


def test_export_is_blind_and_agreement_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDE_RAG_HOME", str(tmp_path))
    ds = tmp_path / "data" / "beir" / "scifact"
    (ds / "qrels").mkdir(parents=True)
    (ds / "corpus.jsonl").write_text(
        json.dumps({"_id": "d1", "title": "T", "text": "Vitamin D cut fractures."}) + "\n"
    )
    (ds / "queries.jsonl").write_text(json.dumps({"_id": "1", "text": "claim"}) + "\n")
    (ds / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\n1\td1\t1\n")
    res = tmp_path / "results" / "gen"
    res.mkdir(parents=True)
    claims = [
        {"sentence": 0, "claim": f"c{i}", "verdict": "supported" if i % 3 else "not_supported"}
        for i in range(9)
    ]
    item = {
        "system": "s",
        "qid": "1",
        "sentences": [{"text": "Vitamin D cut fractures.", "cited_ids": ["d1:0:0"]}],
        "judge_claims": claims,
    }
    (res / "items.jsonl").write_text(json.dumps(item) + "\n")

    sheet = jv.export("gen", "scifact", n=6, seed=1)
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig", newline="")))
    assert (
        len(rows) == 6 and "judge" not in rows[0] and "Vitamin D cut fractures." in rows[0]["cited_passages"]
    )
    key = json.loads((sheet.parent / "gen_judge_key.json").read_text())
    assert sum(v == "not_supported" for v in key.values()) == 3  # balanced
    # A human who always agrees with the judge -> agreement 1, kappa 1.
    for r in rows:
        r["human_label"] = key[r["item_id"]].replace("_", " ")
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=jv.COLUMNS)
        w.writeheader()
        w.writerows(rows)
    result = jv.agreement("gen")
    assert result["agreement"] == 1.0 and result["cohen_kappa"] == 1.0 and result["n_labelled"] == 6
    assert (res / "judge_validation.json").exists()


def test_corrected_faithfulness_reweights_by_human_agreement():
    from hyde_rag.eval.judge_validation import corrected_faithfulness

    v = {"human_supported_rate_given_judge": {"supported": 1.0, "not_supported": 0.8}}
    assert corrected_faithfulness(0.9, v) == pytest.approx(0.9 * 1.0 + 0.1 * 0.8)
    assert corrected_faithfulness(0.9, {}) is None
