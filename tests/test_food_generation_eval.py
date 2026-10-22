"""Food answer evaluation: sampling, the gate, grading and abstention scoring, offline (cached LLM output)."""

import json

import pytest

from hyde_rag.eval import food_generation_eval as fge
from hyde_rag.eval.food_generation_eval import FoodGenerationEval, FoodItem, sample_items
from hyde_rag.eval.judge import correctness_request, faithfulness_request
from hyde_rag.generation.answer import build_request, parse_answer
from hyde_rag.ingest.chunking import DocChunk
from hyde_rag.llm.cache import JsonlCache, cache_key
from hyde_rag.llm.client import LLMResponse, default_base_url
from hyde_rag.retrieval.hyde import cache_path


def _item(i, t, answerable=True):
    return FoodItem(
        f"{t}-{i}", f"question {t} {i}?", t, answerable, "ref" if answerable else "NOT ANSWERABLE"
    )


def test_sample_keeps_every_unanswerable_and_stratifies_the_rest():
    items = [_item(i, "factual") for i in range(10)] + [_item(i, "numeric") for i in range(10)]
    items += [_item(i, "unanswerable", answerable=False) for i in range(3)]
    s = sample_items(items, n=10, seed=13)
    assert len(s) == 10
    assert sum(not it.answerable for it in s) == 3
    types = [it.type for it in s if it.answerable]
    assert abs(types.count("factual") - types.count("numeric")) <= 1
    assert [it.id for it in sample_items(items, n=10, seed=13)] == [it.id for it in s]  # deterministic
    assert len(sample_items(items, n=100, seed=13)) == len(items)


CHUNKS = {
    "fc:1:0": DocChunk(
        "fc:1:0", "fc", 1, 0, "Cooling", "Cool cooked food from 57C to 21C within 2 hours.", 0, 48
    ),
    "fc:2:0": DocChunk("fc:2:0", "fc", 2, 0, "Holding", "Hold hot food at 57C or above.", 48, 78),
    "fc:3:0": DocChunk("fc:3:0", "fc", 3, 0, "Misc", "Unrelated text about plumbing.", 78, 108),
}


@pytest.fixture()
def ev(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDE_RAG_HOME", str(tmp_path))
    test = [
        FoodItem("q1", "How fast must cooked food cool?", "numeric", True, "From 57C to 21C within 2 hours."),
        FoodItem("q2", "Hot holding temperature?", "jargon", True, "57C or above."),
        FoodItem("q3", "What is the fine for plumbing faults?", "unanswerable", False, "NOT ANSWERABLE"),
    ]
    dev = [
        FoodItem("d1", "cooling time?", "numeric", True, "2 hours"),
        FoodItem("d2", "holding temperature?", "jargon", True, "57C"),
        FoodItem("d3", "plumbing fine?", "unanswerable", False, "NOT ANSWERABLE"),
    ]
    runs = {
        "test": {
            "q1": {"fc:1:0": 0.9, "fc:2:0": 0.5},
            "q2": {"fc:2:0": 0.8, "fc:1:0": 0.4},
            "q3": {"fc:3:0": 0.2},  # only off-topic text retrieved
        },
        "dev": {"d1": {"fc:1:0": 0.9}, "d2": {"fc:2:0": 0.9}, "d3": {"fc:3:0": 0.2}},
    }
    monkeypatch.setattr(fge, "load_items", lambda split: dev if split == "dev" else test)
    monkeypatch.setattr(fge, "_load_run", lambda exp, system: runs["dev" if exp == "food-dev" else "test"])
    e = object.__new__(FoodGenerationEval)
    e.cfg = {
        "retrieval_experiment": "food-test",
        "systems": ["dense-bge"],
        "gate": {"experiment": "food-dev", "system": "same", "max_false_refusal": 0.0},
    }
    e.name, e.top_k, e.settings = "food-generation", 2, {}
    e.out_dir = tmp_path / "results" / "food-generation"
    e.answerer, e.judge = "fake/answerer", "fake/judge"
    e.items = {it.id: it for it in test}
    e.sample = test
    e.chunks_by_id = CHUNKS
    e.titles = {"fc": "Food Code"}
    # Gate scores: high when the chunk is about cooling or holding, low for plumbing.
    e._gate_scores = lambda qid, q, chunks: [5.0 if "plumbing" not in c.text else -5.0 for c in chunks]
    return e


def test_food_eval_scores_answers_abstention_and_grades(ev):
    run = fge._load_run("food-test", "dense-bge")
    base = default_base_url()
    ans = JsonlCache(cache_path("food", "answers", "dense-bge", ev.answerer, "food"))
    judge = JsonlCache(cache_path("food", "judge", "dense-bge", ev.judge, "food"))
    corr = JsonlCache(cache_path("food", "correctness", "dense-bge", ev.judge, "food"))
    texts = {
        "q1": "Cooked food must cool from 57C to 21C within 2 hours. [fc:1:0]",
        "q2": "Not found in the provided documents.",  # wrongly abstains on an answerable question
    }
    for q, text in texts.items():
        it = ev.items[q]
        chunks = ev.chunks(run, q, it.question)
        req = build_request(it.question, chunks, ev.answerer)
        ans.put(cache_key(base, req, 0), q, 0, LLMResponse(text, "stop", ev.answerer, 100, 20, 1.0))
    a1 = parse_answer(texts["q1"], ev.chunks(run, "q1", ev.items["q1"].question))
    verdict = {
        "claims": [{"claim": "cool within 2 h", "verdict": "supported"}],
        "citations": [{"sentence": 0, "passage_id": "fc:1:0", "supports": True}],
    }
    judge.put(
        cache_key(base, faithfulness_request(a1, ev.judge), 0),
        "q1",
        0,
        LLMResponse(json.dumps(verdict), "stop", ev.judge, 50, 10, 0.5),
    )
    corr_req = correctness_request(
        ev.items["q1"].question, a1.text, ev.items["q1"].reference_answer, ev.judge
    )
    corr.put(
        cache_key(base, corr_req, 0),
        "q1",
        0,
        LLMResponse(json.dumps({"verdict": "correct", "reason": "matches"}), "stop", ev.judge, 40, 8, 0.4),
    )

    summary = ev.run(llm=False, log=lambda m: None)
    m = summary["systems"]["dense-bge"]
    assert summary["gates"]["dense-bge"]["dev_unanswerable"] == 1
    assert m["complete"] and m["n_answered"] == 3
    assert m["gated"] == 1  # q3: plumbing chunks fall below tau, no LLM call
    assert m["abstention_recall"] == 1.0 and m["abstention_precision"] == 0.5  # q2 abstained wrongly
    assert m["false_refusal_rate"] == 0.5
    assert m["correctness"] == pytest.approx(0.5)  # q1 correct (1.0), q2 abstained (0.0)
    assert m["correctness_by_type"] == {"jargon": 0.0, "numeric": 1.0}
    assert m["faithfulness"] == 1.0 and m["citation_validity"] == 1.0
    items = [
        json.loads(line) for line in (ev.out_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {r["qid"]: r["grade"] for r in items} == {"q1": "correct", "q2": "incorrect", "q3": None}


def test_stage_runs_write_nothing(ev):
    ev.run(llm=False, log=lambda m: None, stage="judge")
    assert not (ev.out_dir / "metrics.json").exists()
