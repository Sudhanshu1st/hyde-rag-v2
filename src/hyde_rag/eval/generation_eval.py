"""Generation evaluation on SciFact: verdicts, citations, abstention and faithfulness.

For each retrieval system (its saved run in results/<retrieval_experiment>/runs/):
  1. take the top-k abstracts for each sampled claim (chunk id = doc_id:0:0);
  2. layer (a): abstain without an LLM call if the top rerank score < tau (tau from dev only);
  3. otherwise ask the answerer for a verdict with inline citations (layer (b) prompt rule);
  4. validate citations; judge claim-level faithfulness and citation precision;
  5. score verdicts and abstention against the gold SciFact labels.
All LLM calls go through the free-tier runner and its committed cache.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from hyde_rag import paths
from hyde_rag.eval.judge import faithfulness_request, parse_faithfulness
from hyde_rag.eval.stats import bootstrap_ci
from hyde_rag.generation.abstention import calibrate_tau
from hyde_rag.generation.answer import Answer, build_request, gate, gated_answer, parse_answer
from hyde_rag.generation.context import Chunk
from hyde_rag.ingest.beir import load_beir
from hyde_rag.ingest.scifact_labels import GOLD_VERDICT, load_labels, stratified_sample
from hyde_rag.llm.cache import JsonlCache, cache_key
from hyde_rag.llm.client import default_base_url
from hyde_rag.llm.quota import Limits, load_limits
from hyde_rag.llm.runner import WorkItem, run_bulk
from hyde_rag.retrieval.hyde import cache_path

VERDICTS = ("SUPPORTED", "REFUTED", "NOT FOUND")


@dataclass
class ItemResult:
    qid: str
    gold: str
    verdict: str | None
    abstained: bool
    gated: bool
    citations_valid: bool | None
    faithfulness: float | None = None
    citation_precision: float | None = None
    n_claims: int = 0
    tokens: int = 0
    latency_s: float | None = None


def _load_run(experiment: str, system: str) -> dict[str, dict[str, float]]:
    path = paths.results_dir() / experiment / "runs" / f"{system}.json.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run: uv run hyde-rag run configs/experiments/{experiment}.yaml"
        )
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


class GenerationEval:
    def __init__(self, config_path: Path):
        self.cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.name = self.cfg["name"]
        self.ds = load_beir(self.cfg["dataset"], split="test")
        self.labels = load_labels("test")
        sample = self.cfg["sample"]
        self.sample = stratified_sample(self.labels, sample["n"], sample["seed"])
        llm = yaml.safe_load((paths.CONFIG_DIR / "llm.yaml").read_text(encoding="utf-8"))
        self.answerer = self.cfg.get("answerer") or llm["roles"]["answerer"]
        self.judge = self.cfg.get("judge") or llm["roles"]["judge"]
        if self.answerer == self.judge:
            raise ValueError("The judge must be a different model from the answerer.")
        self.settings = llm.get("generation", {})
        self.top_k = int(self.cfg.get("top_k", 3))
        self.out_dir = paths.results_dir() / self.name

    def chunks(self, run: dict[str, dict[str, float]], qid: str) -> list[Chunk]:
        top = sorted(run.get(qid, {}).items(), key=lambda kv: (-kv[1], kv[0]))[: self.top_k]
        return [
            Chunk(did, 0, 0, self.ds.corpus[did].text, title=self.ds.corpus[did].title, score=score)
            for did, score in top
        ]

    def calibrate(self) -> dict:
        g = self.cfg["gate"]
        dev = _load_run(g["experiment"], g["system"])
        train = load_labels("train")
        ans = [max(dev[q].values()) for q in dev if dev[q] and train.get(q) in ("SUPPORT", "CONTRADICT")]
        una = [max(dev[q].values()) for q in dev if dev[q] and train.get(q) == "NONE"]
        cal = calibrate_tau(ans, una, max_false_refusal=float(g.get("max_false_refusal", 0.05)))
        return {**asdict(cal), "dev_answerable": len(ans), "dev_unanswerable": len(una)}

    def _reasoning(self, model: str) -> str | None:
        return self.settings.get(model, {}).get("reasoning_effort")

    def run(self, *, llm: bool = True, approved: bool = False, log=print) -> dict:
        gate_info = self.calibrate()
        tau = gate_info["tau"]
        log(f"gate: tau={tau} (dev F1={gate_info['f1']:.2f}, false refusal={gate_info['false_refusal']:.2f})")
        client = None
        if llm:
            from hyde_rag.llm.client import OpenAICompatClient

            client = OpenAICompatClient()
        base_url = client.base_url if client else default_base_url()
        limits = load_limits()
        unlimited = Limits(rpm=10**6, rpd=10**9, tpm=10**12, tpd=10**15)

        summary: dict = {
            "experiment": self.name,
            "n_claims": len(self.sample),
            "top_k": self.top_k,
            "answerer": self.answerer,
            "judge": self.judge,
            "gate": gate_info,
            "label_mix": dict(Counter(self.labels[q] for q in self.sample)),
            "systems": {},
        }
        for system in self.cfg["systems"]:
            try:
                run = _load_run(self.cfg["retrieval_experiment"], system)
            except FileNotFoundError as e:
                log(f"- {system}: skipped ({e})")
                continue
            if any(q not in run for q in self.sample):
                log(f"- {system}: skipped (run lacks some sampled claims)")
                continue

            answers: dict[str, Answer] = {}
            items: list[tuple[WorkItem, list[Chunk]]] = []
            for q in self.sample:
                chunks = self.chunks(run, q)
                if gate(chunks, tau):
                    answers[q] = gated_answer(chunks, "claim")
                    continue
                req = build_request(
                    self.ds.queries[q],
                    chunks,
                    self.answerer,
                    mode="claim",
                    reasoning_effort=self._reasoning(self.answerer),
                )
                items.append((WorkItem(q, 0, req), chunks))

            ans_cache = JsonlCache(cache_path(self.ds.name, "answers", system, self.answerer))
            if client and items:
                run_bulk(
                    client,
                    ans_cache,
                    [it for it, _ in items],
                    limits.get(self.answerer, unlimited),
                    approved=approved,
                    log=log,
                )
            usage: dict[str, tuple[int, float]] = {}
            for it, chunks in items:
                rec = ans_cache.get(cache_key(base_url, it.request, 0))
                if rec is not None:
                    answers[it.item_id] = parse_answer(rec.text, chunks, "claim")
                    usage[it.item_id] = (rec.prompt_tokens + rec.completion_tokens, rec.latency_s)

            judged = {q: a for q, a in answers.items() if not a.abstained and a.citations.sentences}
            judge_items = [
                WorkItem(q, 0, faithfulness_request(a, self.judge, self._reasoning(self.judge)))
                for q, a in sorted(judged.items())
            ]
            judge_cache = JsonlCache(cache_path(self.ds.name, "judge", system, self.judge))
            if client and judge_items:
                run_bulk(
                    client,
                    judge_cache,
                    judge_items,
                    limits.get(self.judge, unlimited),
                    approved=approved,
                    log=log,
                )

            results: list[ItemResult] = []
            for q in self.sample:
                if q not in answers:
                    continue  # not generated yet
                a = answers[q]
                r = ItemResult(
                    q,
                    self.labels[q],
                    a.verdict,
                    a.abstained,
                    a.gated,
                    None if a.abstained else a.citations.valid,
                )
                if q in usage:
                    r.tokens, r.latency_s = usage[q]
                if q in judged:
                    wi = next(w for w in judge_items if w.item_id == q)
                    rec = judge_cache.get(cache_key(base_url, wi.request, 0))
                    if rec is not None:
                        try:
                            f = parse_faithfulness(rec.text, a)
                            r.faithfulness, r.citation_precision, r.n_claims = (
                                f.faithfulness,
                                f.citation_precision,
                                f.n_claims,
                            )
                        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                            pass
                results.append(r)
            summary["systems"][system] = self._metrics(results)
            m = summary["systems"][system]
            log(
                f"- {system}: {m['n_answered']}/{len(self.sample)} answered; accuracy={m['verdict_accuracy']}"
            )

        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _metrics(self, results: list[ItemResult]) -> dict:
        n = len(results)
        out: dict = {"n_answered": n, "complete": n == len(self.sample)}
        if not n:
            return out | {"verdict_accuracy": None}
        correct = {r.qid: float(r.verdict == GOLD_VERDICT[r.gold]) for r in results}
        acc, lo, hi = bootstrap_ci(correct)
        f1s = []
        for v in VERDICTS:
            tp = sum(1 for r in results if r.verdict == v and GOLD_VERDICT[r.gold] == v)
            pred = sum(1 for r in results if r.verdict == v)
            gold = sum(1 for r in results if GOLD_VERDICT[r.gold] == v)
            p, rc = (tp / pred if pred else 0.0), (tp / gold if gold else 0.0)
            f1s.append(2 * p * rc / (p + rc) if p + rc else 0.0)
        should = [r for r in results if r.gold == "NONE"]
        answerable = [r for r in results if r.gold != "NONE"]
        abstained = [r for r in results if r.abstained]
        answered = [r for r in results if not r.abstained]
        faith = {r.qid: r.faithfulness for r in answered if r.faithfulness is not None}
        cites = [r for r in answered if r.citation_precision is not None]
        out |= {
            "verdict_accuracy": acc,
            "verdict_accuracy_ci95": [lo, hi],
            "verdict_macro_f1": float(np.mean(f1s)),
            "abstention_precision": sum(r.gold == "NONE" for r in abstained) / len(abstained)
            if abstained
            else None,
            "abstention_recall": sum(r.abstained for r in should) / len(should) if should else None,
            "false_refusal_rate": sum(r.abstained for r in answerable) / len(answerable)
            if answerable
            else None,
            "gated": sum(r.gated for r in results),
            "citation_validity": (
                sum(bool(r.citations_valid) for r in answered) / len(answered) if answered else None
            ),
            "faithfulness": bootstrap_ci(faith)[0] if faith else None,
            "faithfulness_ci95": list(bootstrap_ci(faith)[1:]) if faith else None,
            "n_judged": len(faith),
            "citation_precision": float(np.mean([r.citation_precision for r in cites])) if cites else None,
            "tokens_per_answer": float(np.mean([r.tokens for r in answered if r.tokens]))
            if answered
            else None,
            "latency_s": float(np.median([r.latency_s for r in answered if r.latency_s]))
            if answered
            else None,
        }
        return out
