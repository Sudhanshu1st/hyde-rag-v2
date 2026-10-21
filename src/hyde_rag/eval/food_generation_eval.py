"""Grounded-answer evaluation on the food-safety eval set (question answering, not claims).

For each retrieval system (its saved run in results/<retrieval_experiment>/runs/):
  1. take the top-k chunks for each sampled test question;
  2. layer (a): abstain without an LLM call if the best cross-encoder score < tau, where tau is
     calibrated on the food dev split (questions and answerability only, never the test split);
  3. otherwise ask the answerer for an answer with inline citations (layer (b) prompt rule);
  4. validate citations; judge claim-level faithfulness and citation precision; grade the
     answer against the reviewed reference answer (correct / partially correct / incorrect);
  5. score abstention against the answerable flag (unanswerable questions should abstain).
Same answer prompt, gate, judge and free-tier runner as the SciFact evaluation and the API.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from hyde_rag import paths
from hyde_rag.eval.judge import (
    correctness_request,
    faithfulness_request,
    parse_correctness,
    parse_faithfulness,
)
from hyde_rag.eval.stats import bootstrap_ci
from hyde_rag.generation.abstention import calibrate_tau
from hyde_rag.generation.answer import Answer, build_request, gate, gated_answer, parse_answer
from hyde_rag.generation.context import Chunk
from hyde_rag.llm.cache import JsonlCache, cache_key
from hyde_rag.llm.client import default_base_url
from hyde_rag.llm.quota import Limits, load_limits
from hyde_rag.llm.runner import WorkItem, run_bulk
from hyde_rag.retrieval.hyde import cache_path

EVALSET_DIR = paths.REPO_ROOT / "eval_sets" / "food_quality"
GRADE_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


@dataclass
class FoodItem:
    id: str
    question: str
    type: str
    answerable: bool
    reference_answer: str


@dataclass
class ItemResult:
    qid: str
    type: str
    answerable: bool
    abstained: bool
    gated: bool
    citations_valid: bool | None
    grade: str | None = None  # correct / partially_correct / incorrect (answerable items)
    faithfulness: float | None = None
    citation_precision: float | None = None
    n_claims: int = 0
    tokens: int = 0
    latency_s: float | None = None


def load_items(split: str) -> list[FoodItem]:
    path = EVALSET_DIR / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [
        FoodItem(r["id"], r["question"], r["type"], bool(r["answerable"]), r["reference_answer"])
        for r in rows
    ]


def sample_items(items: list[FoodItem], n: int, seed: int) -> list[FoodItem]:
    """Every unanswerable item (abstention needs all the support it can get), then answerable items
    stratified by type in proportion to the split, up to n in total. Deterministic."""
    if n >= len(items):
        return sorted(items, key=lambda it: it.id)
    rng = random.Random(seed)
    una = [it for it in items if not it.answerable]
    by_type: dict[str, list[FoodItem]] = defaultdict(list)
    for it in sorted(items, key=lambda it: it.id):
        if it.answerable:
            by_type[it.type].append(it)
    budget = max(0, n - len(una))
    total = sum(len(v) for v in by_type.values())
    chosen: list[FoodItem] = []
    quotas = {t: int(budget * len(v) / total) for t, v in by_type.items()}
    # Hand out the rounding remainder to the largest types first.
    for t in sorted(by_type, key=lambda t: -len(by_type[t]))[: budget - sum(quotas.values())]:
        quotas[t] += 1
    for t, group in sorted(by_type.items()):
        chosen += rng.sample(group, min(quotas[t], len(group)))
    return sorted(una + chosen, key=lambda it: it.id)


def _load_run(experiment: str, system: str) -> dict[str, dict[str, float]]:
    path = paths.results_dir() / experiment / "runs" / f"{system}.json.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run: uv run hyde-rag run configs/experiments/{experiment}.yaml"
        )
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


class FoodGenerationEval:
    def __init__(self, config_path: Path):
        from hyde_rag.ingest.food import load_chunks, load_docs

        self.cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.name = self.cfg["name"]
        self.items = {it.id: it for it in load_items("test")}
        s = self.cfg["sample"]
        self.sample = sample_items(list(self.items.values()), int(s["n"]), int(s["seed"]))
        self.chunks_by_id = {c.chunk_id: c for c in load_chunks()}
        self.titles = {d: v["title"] for d, v in load_docs().items()}
        llm = yaml.safe_load((paths.CONFIG_DIR / "llm.yaml").read_text(encoding="utf-8"))
        self.answerer = self.cfg.get("answerer") or llm["roles"]["answerer"]
        self.judge = self.cfg.get("judge") or llm["roles"]["judge"]
        if self.answerer == self.judge:
            raise ValueError("The judge must be a different model from the answerer.")
        self.settings = llm.get("generation", {})
        self.top_k = int(self.cfg.get("top_k", 3))
        self.out_dir = paths.results_dir() / self.name

    # ---- retrieval context and the abstention gate ----
    def _gate_scores(self, qid: str, question: str, chunks: list[Chunk]) -> list[float]:
        """Cross-encoder (question, chunk text) scores, as the live pipeline computes them."""
        if not hasattr(self, "_ce"):
            from hyde_rag.retrieval.rerank import DEFAULT_RERANKER, CrossEncoderReranker

            cache = paths.index_dir() / "food" / f"gate__{DEFAULT_RERANKER.replace('/', '__')}.tsv"
            self._ce = CrossEncoderReranker(DEFAULT_RERANKER, cache_file=cache, threads=4)
        return self._ce.score([(qid, c.chunk_id, question, c.text) for c in chunks])

    def chunks(self, run: dict[str, dict[str, float]], qid: str, question: str) -> list[Chunk]:
        top = sorted(run.get(qid, {}).items(), key=lambda kv: (-kv[1], kv[0]))[: self.top_k]
        raw = [self.chunks_by_id[cid] for cid, _ in top]
        plain = [Chunk(c.doc_id, c.page, c.chunk, c.text) for c in raw]
        scores = self._gate_scores(qid, question, plain)
        return [
            Chunk(c.doc_id, c.page, c.chunk, c.text, title=self.titles[c.doc_id], section=c.section, score=s)
            for c, s in zip(raw, scores, strict=True)
        ]

    def calibrate(self, system: str) -> dict:
        """tau from the dev split. gate.system "same" uses each evaluated system's own dev run
        (a reranked list has systematically higher top scores than a dense one)."""
        g = self.cfg["gate"]
        dev_run = _load_run(g["experiment"], system if g["system"] == "same" else g["system"])
        dev = load_items("dev")
        best = {it.id: max(c.score for c in self.chunks(dev_run, it.id, it.question)) for it in dev}
        ans = [best[it.id] for it in dev if it.answerable]
        una = [best[it.id] for it in dev if not it.answerable]
        cal = calibrate_tau(ans, una, max_false_refusal=float(g.get("max_false_refusal", 0.05)))
        return {
            **asdict(cal),
            "dev_answerable": len(ans),
            "dev_unanswerable": len(una),
            "scorer": "cross-encoder",
            "dev_system": system if g["system"] == "same" else g["system"],
        }

    def _reasoning(self, model: str) -> str | None:
        return self.settings.get(model, {}).get("reasoning_effort")

    # ---- the evaluation ----
    def run(self, *, llm: bool = True, approved: bool = False, log=print) -> dict:
        client = None
        if llm:
            from hyde_rag.llm.client import OpenAICompatClient

            client = OpenAICompatClient()
        base_url = client.base_url if client else default_base_url()
        limits = load_limits()
        unlimited = Limits(rpm=10**6, rpd=10**9, tpm=10**12, tpd=10**15)

        def bulk(cache: JsonlCache, work: list[WorkItem], model: str) -> None:
            if client and work:
                run_bulk(client, cache, work, limits.get(model, unlimited), approved=approved, log=log)

        summary: dict = {
            "experiment": self.name,
            "n_questions": len(self.sample),
            "top_k": self.top_k,
            "answerer": self.answerer,
            "judge": self.judge,
            "gates": {},
            "type_mix": dict(Counter(it.type for it in self.sample)),
            "systems": {},
        }
        item_records: list[dict] = []
        for system in self.cfg["systems"]:
            try:
                run = _load_run(self.cfg["retrieval_experiment"], system)
            except FileNotFoundError as e:
                log(f"- {system}: skipped ({e})")
                continue
            if any(it.id not in run for it in self.sample):
                log(f"- {system}: skipped (run lacks some sampled questions)")
                continue
            summary["gates"][system] = g = self.calibrate(system)
            tau, f1, fr = g["tau"], g["f1"], g["false_refusal"]
            log(f"- {system}: gate tau={tau:.3f}, dev F1={f1:.2f}, false refusals={fr:.2f}")

            # 1. Answers (gated items need no call).
            answers: dict[str, Answer] = {}
            work: list[tuple[WorkItem, list[Chunk]]] = []
            for it in self.sample:
                chunks = self.chunks(run, it.id, it.question)
                if gate(chunks, tau):
                    answers[it.id] = gated_answer(chunks)
                    continue
                req = build_request(
                    it.question, chunks, self.answerer, reasoning_effort=self._reasoning(self.answerer)
                )
                work.append((WorkItem(it.id, 0, req), chunks))
            ans_cache = JsonlCache(cache_path("food", "answers", system, self.answerer, "food"))
            bulk(ans_cache, [w for w, _ in work], self.answerer)
            usage: dict[str, tuple[int, float]] = {}
            for w, chunks in work:
                rec = ans_cache.get(cache_key(base_url, w.request, 0))
                if rec is not None:
                    answers[w.item_id] = parse_answer(rec.text, chunks)
                    usage[w.item_id] = (rec.prompt_tokens + rec.completion_tokens, rec.latency_s)

            # 2. Judge: faithfulness for every cited answer, correctness for answered answerable items.
            faith_work = {
                q: WorkItem(q, 0, faithfulness_request(a, self.judge, self._reasoning(self.judge)))
                for q, a in sorted(answers.items())
                if not a.abstained and a.citations.sentences
            }
            corr_work = {
                q: WorkItem(
                    q,
                    0,
                    correctness_request(
                        self.items[q].question,
                        a.text,
                        self.items[q].reference_answer,
                        self.judge,
                        self._reasoning(self.judge),
                    ),
                )
                for q, a in sorted(answers.items())
                if not a.abstained and self.items[q].answerable
            }
            faith_cache = JsonlCache(cache_path("food", "judge", system, self.judge, "food"))
            corr_cache = JsonlCache(cache_path("food", "correctness", system, self.judge, "food"))
            bulk(faith_cache, list(faith_work.values()), self.judge)
            bulk(corr_cache, list(corr_work.values()), self.judge)

            # 3. Score.
            results: list[ItemResult] = []
            for it in self.sample:
                a = answers.get(it.id)
                if a is None:
                    continue  # not generated yet
                r = ItemResult(
                    it.id,
                    it.type,
                    it.answerable,
                    a.abstained,
                    a.gated,
                    None if a.abstained else a.citations.valid,
                )
                if it.id in usage:
                    r.tokens, r.latency_s = usage[it.id]
                if it.answerable and a.abstained:
                    r.grade = "incorrect"  # abstaining on an answerable question earns nothing
                elif it.id in corr_work:
                    rec = corr_cache.get(cache_key(base_url, corr_work[it.id].request, 0))
                    if rec is not None:
                        with contextlib.suppress(json.JSONDecodeError, KeyError, TypeError):
                            r.grade = parse_correctness(rec.text)
                claims: list[dict] = []
                if it.id in faith_work:
                    rec = faith_cache.get(cache_key(base_url, faith_work[it.id].request, 0))
                    if rec is not None:
                        try:
                            f = parse_faithfulness(rec.text, a)
                            r.faithfulness, r.citation_precision, r.n_claims = (
                                f.faithfulness,
                                f.citation_precision,
                                f.n_claims,
                            )
                            claims = json.loads(rec.text).get("claims", [])
                        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                            pass
                results.append(r)
                # The model's own text and chunk IDs only, never corpus text.
                item_records.append(
                    {
                        "system": system,
                        **asdict(r),
                        "answer": a.text,
                        "retrieved": [c.chunk_id for c in a.chunks],
                        "gate_scores": [c.score for c in a.chunks],
                        "sentences": [
                            {"text": s.text, "cited_ids": s.cited_ids} for s in a.citations.sentences
                        ],
                        "judge_claims": claims,
                    }
                )
            summary["systems"][system] = m = self._metrics(results)
            log(f"- {system}: {m['n_answered']}/{len(self.sample)} done; correctness={m['correctness']}")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (self.out_dir / "items.jsonl").open("w", encoding="utf-8") as f:
            for rec in item_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return summary

    def _metrics(self, results: list[ItemResult]) -> dict:
        out: dict = {"n_answered": len(results), "complete": len(results) == len(self.sample)}
        answerable = [r for r in results if r.answerable]
        graded = {r.qid: GRADE_SCORE[r.grade] for r in answerable if r.grade in GRADE_SCORE}
        strict = {r.qid: float(r.grade == "correct") for r in answerable if r.grade in GRADE_SCORE}
        should = [r for r in results if not r.answerable]
        abstained = [r for r in results if r.abstained]
        answered = [r for r in results if not r.abstained]
        faith = {r.qid: r.faithfulness for r in answered if r.faithfulness is not None}
        cites = [r for r in answered if r.citation_precision is not None]
        by_type: dict[str, float] = {}
        for t in sorted({r.type for r in answerable}):
            vals = [GRADE_SCORE[r.grade] for r in answerable if r.type == t and r.grade in GRADE_SCORE]
            by_type[t] = float(np.mean(vals)) if vals else None
        out |= {
            # correct = 1, partially correct = 0.5, incorrect or abstained = 0 (answerable items)
            "correctness": bootstrap_ci(graded)[0] if graded else None,
            "correctness_ci95": list(bootstrap_ci(graded)[1:]) if graded else None,
            "correct_rate": float(np.mean(list(strict.values()))) if strict else None,
            "n_graded": len(graded),
            "correctness_by_type": by_type,
            "abstention_precision": sum(not r.answerable for r in abstained) / len(abstained)
            if abstained
            else None,
            "abstention_recall": sum(r.abstained for r in should) / len(should) if should else None,
            "false_refusal_rate": sum(r.abstained for r in answerable) / len(answerable)
            if answerable
            else None,
            "gated": sum(r.gated for r in results),
            "citation_validity": sum(bool(r.citations_valid) for r in answered) / len(answered)
            if answered
            else None,
            "faithfulness": bootstrap_ci(faith)[0] if faith else None,
            "faithfulness_ci95": list(bootstrap_ci(faith)[1:]) if faith else None,
            "n_judged": len(faith),
            "citation_precision": float(np.mean([r.citation_precision for r in cites])) if cites else None,
            "tokens_per_answer": float(np.mean([r.tokens for r in answered if r.tokens]))
            if any(r.tokens for r in answered)
            else None,
            "latency_s": float(np.median([r.latency_s for r in answered if r.latency_s]))
            if any(r.latency_s for r in answered)
            else None,
        }
        return out
