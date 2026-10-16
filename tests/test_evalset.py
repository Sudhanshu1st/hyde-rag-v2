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


def test_review_csv_merge_keeps_reviewer_work(tmp_path):
    from hyde_rag.eval.question_gen import Candidate, stable_id, write_review_csv

    assert stable_id("factual", "factual|doc:1:0", 0) == stable_id("factual", "factual|doc:1:0", 0)
    ev = [{"doc_id": "d", "page": 1, "quote": "q", "located": True}]
    a = Candidate("fac-a-0", "factual", "Q1?", "A1", True, ev, ["d:1:0"])
    b = Candidate("una-b-0", "unanswerable", "Q2?", "NOT ANSWERABLE", False, [], ["d:2:0"])
    path = tmp_path / "review.csv"
    write_review_csv([a, b], path, {"una-b-0": "[d:9:0] closest text"})
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    assert [r["id"] for r in rows[:2]] == ["fac-a-0", "una-b-0"] and len(rows) == 32  # + 30 human rows
    assert rows[1]["check_passages"].startswith("[d:9:0]")
    # Reviewer fills in a decision and writes a question of their own.
    rows[0]["decision"], rows[0]["edited_question"] = "edit", "Better Q1?"
    rows[2]["question"], rows[2]["answerable"] = "My own question?", "yes"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # A later batch adds a new candidate; earlier work must survive.
    c = Candidate("num-c-0", "numeric", "Q3?", "5C", True, ev, ["e:1:0"])
    write_review_csv([c], path)
    after = {r["id"]: r for r in csv.DictReader(path.open(encoding="utf-8-sig", newline=""))}
    assert after["fac-a-0"]["decision"] == "edit" and after["fac-a-0"]["edited_question"] == "Better Q1?"
    assert "una-b-0" in after and "num-c-0" in after
    assert after["hum-001"]["question"] == "My own question?"
    assert sum(1 for r in after.values() if r["type"] == "human") == 30


def test_sampler_v2_spreads_questions_across_documents():
    from hyde_rag.eval.question_gen import sample_chunks
    from hyde_rag.ingest.chunking import DocChunk

    words = " ".join(["requirement"] * 90)
    chunks = []
    for d in range(6):
        for k in range(10):
            # Only documents 0-1 have numeric limits; the others must still get other question types.
            numeric = " cool to 5 \N{DEGREE SIGN}C within 4 hours" if d < 2 else ""
            chunks.append(DocChunk(f"d{d}:1:{k}", f"d{d}", 1, k, "s", words + numeric, 0, 1))
    v2 = sample_chunks(chunks, seed=1, scale=0.1, sampler=2)
    v1 = sample_chunks(chunks, seed=1, scale=0.1, sampler=1)
    docs_v2 = {c.doc_id for c in v2["factual"] + v2["jargon"]}
    docs_v1 = {c.doc_id for c in v1["factual"] + v1["jargon"]}
    assert len(docs_v2) >= 4  # fixed: non-numeric documents still get questions
    assert docs_v1 <= {"d0", "d1"}  # the original bug, kept only to rebuild batch "main"
