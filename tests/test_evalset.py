"""Reviewed CSV -> dev/test JSONL, with decisions, edits, human rows and evidence relocation."""

import csv
import json

from hyde_rag.eval import evalset
from hyde_rag.eval.question_gen import REVIEW_COLUMNS

TEXT = (
    "3-501.14 Cooling. Cooked food shall be cooled from 57C to 21C within 2 hours. "
    "Then from 21C to 5C within a total of 6 hours. Shellstock shall be cooled to 5C within 4 hours."
)


def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDE_RAG_HOME", str(tmp_path))
    monkeypatch.setattr(evalset, "EVALSET_DIR", tmp_path / "eval")
    proc = tmp_path / "data" / "food" / "processed"
    (proc / "text").mkdir(parents=True)
    (proc / "text" / "fc.txt").write_text(TEXT, encoding="utf-8")
    chunk = {
        "chunk_id": "fc:87:0",
        "doc_id": "fc",
        "page": 87,
        "chunk": 0,
        "section": "Cooling",
        "text": TEXT,
        "char_start": 0,
        "char_end": len(TEXT),
    }
    (proc / "chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    (tmp_path / "eval").mkdir()
    s = TEXT.index("Cooked food")
    cand = {
        "id": "num-1",
        "evidence": [
            {
                "doc_id": "fc",
                "page": 87,
                "section": "Cooling",
                "char_start": s,
                "char_end": s + 40,
                "quote": "Cooked food shall be cooled",
                "located": True,
            }
        ],
    }
    (tmp_path / "eval" / "candidates.jsonl").write_text(json.dumps(cand) + "\n", encoding="utf-8")
    rows = [
        {
            "id": "num-1",
            "type": "numeric",
            "answerable": "True",
            "question": "Cooling time 57C to 21C?",
            "reference_answer": "2 hours",
            "evidence_doc": "fc",
            "evidence_quote": "Cooked food shall be cooled",
            "decision": "accept",
        },
        {
            "id": "fac-2",
            "type": "factual",
            "answerable": "True",
            "question": "old",
            "reference_answer": "old",
            "evidence_doc": "fc",
            "evidence_quote": "Shellstock shall be cooled to 5C within 4 hours",
            "decision": "edit",
            "edited_question": "How fast must shellstock be cooled?",
            "edited_answer": "To 5C within 4 hours.",
        },
        {
            "id": "una-3",
            "type": "unanswerable",
            "answerable": "False",
            "question": "Cooling time for soup?",
            "reference_answer": "NOT ANSWERABLE",
            "decision": "accept",
        },
        {
            "id": "fac-4",
            "type": "factual",
            "answerable": "True",
            "question": "bad",
            "reference_answer": "x",
            "decision": "reject",
        },
        {
            "id": "fac-5",
            "type": "factual",
            "answerable": "True",
            "question": "unreviewed",
            "reference_answer": "x",
        },
        {
            "id": "fac-6",
            "type": "factual",
            "answerable": "True",
            "question": "no evidence",
            "reference_answer": "x",
            "evidence_doc": "fc",
            "evidence_quote": "this sentence is not in the document at all",
            "decision": "edit",
        },
        {
            "id": "hum-001",
            "type": "human",
            "answerable": "yes",
            "question": "Total cooling time to 5C?",
            "reference_answer": "6 hours",
            "evidence_doc": "fc",
            "evidence_quote": "Then from 21C to 5C within a total of 6 hours",
        },
        {"id": "hum-002", "type": "human", "question": ""},  # unused blank row
    ]
    with (tmp_path / "eval" / "review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in REVIEW_COLUMNS})


def test_build_evalset(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    report = evalset.build()
    assert report.kept == 4 and report.rejected == 1 and report.unreviewed == 1
    assert report.excluded == ["fac-6"]
    items = {}
    for split in ("dev", "test"):
        for line in (tmp_path / "eval" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            it = json.loads(line)
            items[it["id"]] = it
    assert set(items) == {"num-1", "fac-2", "una-3", "hum-001"}
    assert items["fac-2"]["question"] == "How fast must shellstock be cooled?"
    span = items["fac-2"]["gold_evidence"][0]
    assert TEXT[span["char_start"] : span["char_end"]].startswith("Shellstock shall be cooled")
    assert items["una-3"]["answerable"] is False and items["una-3"]["gold_evidence"] == []
    assert items["hum-001"]["source"] == "human" and items["hum-001"]["type"] == "human"
