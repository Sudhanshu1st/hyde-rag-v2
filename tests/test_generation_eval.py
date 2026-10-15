"""End-to-end generation eval on tiny fixtures with a scripted fake LLM (no network, no key)."""

import gzip
import json

import pytest
import yaml

from hyde_rag.eval import generation_eval as ge
from hyde_rag.llm.client import LLMResponse

CORPUS = {
    "d1": ("Vitamin D trial", "Vitamin D supplementation reduced hip fractures by 20 percent."),
    "d2": ("Sleep study", "Shift work was not associated with obesity in this cohort."),
    "d3": ("Unrelated", "Plant roots absorb water."),
}
# claim id -> (claim text, label in the original release)
CLAIMS = {
    "1": ("Vitamin D reduces fractures.", "SUPPORT"),
    "2": ("Shift work causes obesity.", "CONTRADICT"),
    "3": ("Roots photosynthesise at night.", "NONE"),
    "4": ("Vitamin D prevents hip fractures in adults.", "SUPPORT"),
}


class FakeLLM:
    """Answers by claim; judges every claim as supported."""

    base_url = "https://fake.local/v1"
    calls = 0

    def __init__(self, *a, **k):
        pass

    def complete(self, request):
        FakeLLM.calls += 1
        system, user = request.messages[0][1], request.messages[-1][1]
        if request.json_schema is not None:  # judge
            n_sent = user.count("Sentence ")
            cited = [
                line.split("Passage ")[1].split(":")[0:3] for line in user.splitlines() if "Passage " in line
            ]
            text = json.dumps(
                {
                    "claims": [{"sentence": i, "claim": "c", "verdict": "supported"} for i in range(n_sent)],
                    "citations": [
                        {"sentence": 0, "passage_id": ":".join(c), "supports": True} for c in cited
                    ],
                }
            )
        elif "Vitamin D" in user.split("Claim:")[-1]:
            text = "Verdict: SUPPORTED\nSupplementation reduced hip fractures [d1:0:0]."
        elif "Shift work" in user.split("Claim:")[-1]:
            text = "Verdict: REFUTED\nNo association with obesity was found [d2:0:0]. It was a cohort study."
        else:
            text = "Verdict: NOT FOUND\nNot found in the provided documents."
        assert "only" in system.lower()
        return LLMResponse(text, "stop", request.model, 300, 50, 0.4)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDE_RAG_HOME", str(tmp_path))
    ds = tmp_path / "data" / "beir" / "scifact"
    (ds / "qrels").mkdir(parents=True)
    (ds / "corpus.jsonl").write_text(
        "".join(json.dumps({"_id": d, "title": t, "text": x}) + "\n" for d, (t, x) in CORPUS.items()),
        encoding="utf-8",
    )
    (ds / "queries.jsonl").write_text(
        "".join(json.dumps({"_id": q, "text": c}) + "\n" for q, (c, _) in CLAIMS.items()), encoding="utf-8"
    )
    (ds / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n1\td1\t1\n2\td2\t1\n3\td3\t1\n4\td1\t1\n", encoding="utf-8"
    )
    labels = tmp_path / "data" / "scifact_labels"
    labels.mkdir(parents=True)

    def claims_file(rows):
        return "".join(
            json.dumps(
                {"id": int(q), "claim": c, "evidence": {} if lab == "NONE" else {"d": [{"label": lab}]}}
            )
            + "\n"
            for q, (c, lab) in rows.items()
        )

    (labels / "claims_dev.jsonl").write_text(claims_file(CLAIMS), encoding="utf-8")
    # Dev (train) claims: answerable ones score high, unanswerable low -> a usable gate.
    dev = {str(100 + i): ("c", "SUPPORT" if i < 20 else "NONE") for i in range(30)}
    (labels / "claims_train.jsonl").write_text(claims_file(dev), encoding="utf-8")
    # BEIR train split (dev claims): queries + qrels, as in the real dataset.
    with (ds / "queries.jsonl").open("a", encoding="utf-8") as f:
        for q in dev:
            f.write(json.dumps({"_id": q, "text": f"dev claim {q}"}) + "\n")
    (ds / "qrels" / "train.tsv").write_text(
        "query-id\tcorpus-id\tscore\n" + "".join(f"{q}\td1\t1\n" for q in dev), encoding="utf-8"
    )

    def write_run(experiment, system, run):
        out = tmp_path / "results" / experiment / "runs"
        out.mkdir(parents=True, exist_ok=True)
        with gzip.open(out / f"{system}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(run, f)

    write_run(
        "scifact-dev",
        "gate-sys",
        {q: {"d1": 8.0 if lab == "SUPPORT" else -9.0} for q, (_, lab) in dev.items()},
    )
    test_run = {"1": {"d1": 7.0, "d3": 1.0}, "2": {"d2": 6.0}, "3": {"d3": -10.0}, "4": {"d1": 7.5}}
    write_run("scifact", "sys-a", test_run)
    cfg = {
        "name": "gen-test",
        "dataset": "scifact",
        "retrieval_experiment": "scifact",
        "systems": ["sys-a", "missing-sys"],
        "top_k": 2,
        "sample": {"n": 4, "seed": 1},
        "gate": {"experiment": "scifact-dev", "system": "gate-sys", "max_false_refusal": 0.05},
        "answerer": "fake/answerer",
        "judge": "fake/judge",
    }
    path = tmp_path / "gen.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr("hyde_rag.llm.client.OpenAICompatClient", FakeLLM)
    return path


def test_generation_eval_end_to_end(home):
    logs = []
    summary = ge.GenerationEval(home).run(llm=True, log=logs.append)
    assert summary["gate"]["tau"] is not None  # dev separates answerable from unanswerable
    assert any("missing-sys: skipped" in line for line in logs)
    m = summary["systems"]["sys-a"]
    assert m["complete"] and m["n_answered"] == 4
    # Claim 3 (no evidence) is gated before any LLM call; the other three get correct verdicts.
    assert m["gated"] == 1
    assert m["verdict_accuracy"] == pytest.approx(1.0)
    assert (
        m["abstention_precision"] == 1.0 and m["abstention_recall"] == 1.0 and m["false_refusal_rate"] == 0.0
    )
    # Claim 2's answer has an uncited second sentence -> 2 of 3 answers have valid citations.
    assert m["citation_validity"] == pytest.approx(2 / 3)
    assert m["faithfulness"] == pytest.approx(1.0) and m["n_judged"] == 3

    # Re-scoring from the cache needs no client and makes no calls.
    before = FakeLLM.calls
    again = ge.GenerationEval(home).run(llm=False, log=lambda _: None)
    assert FakeLLM.calls == before
    assert again["systems"]["sys-a"]["verdict_accuracy"] == pytest.approx(1.0)
