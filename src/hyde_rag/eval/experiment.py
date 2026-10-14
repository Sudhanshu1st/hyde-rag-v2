"""Config-driven retrieval experiments: one YAML file per experiment, one entry per system.

System types (configs/experiments/*.yaml):
  bm25          BM25 (Pyserini-flat settings)
  bm25_q2d      BM25 with Query2doc expansion: query x5 + hypothetical documents
  dense         naive dense retrieval with an encoder
  hyde          HyDE (Eq. 8) with an encoder, a generator, an instruction and N
  rrf           Reciprocal Rank Fusion of earlier systems (hybrid search)
  multi_query   RAG-Fusion: original query + K LLM rewrites over BM25 and/or dense legs, RRF
  rerank        cross-encoder rerank of an earlier system's top-k

Systems may refer to earlier systems by name. Every system that needs LLM output reads it
from the committed cache; nothing here calls an API.
"""

from __future__ import annotations

import gzip
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from hyde_rag import paths
from hyde_rag.eval.metrics import evaluate, remove_self_hits
from hyde_rag.eval.stats import bootstrap_ci, compare_paired, holm
from hyde_rag.ingest.beir import load_beir
from hyde_rag.llm.cache import JsonlCache
from hyde_rag.retrieval.bm25 import BM25Retriever, doc_text_flat
from hyde_rag.retrieval.dense import (
    ENCODERS,
    DenseIndex,
    Encoder,
    doc_text_dense,
    index_path,
    load_index,
)
from hyde_rag.retrieval.fusion import hyde_vector, rrf, rrf_runs
from hyde_rag.retrieval.hyde import cache_path, parse_rewrites

Run = dict[str, dict[str, float]]

SAVED_RUN_DEPTH = 100  # committed runs keep the top-100 (enough for every reported metric)
LATENCY_PROBE_QUERIES = 10


@dataclass
class SystemResult:
    name: str
    spec: dict
    run: Run
    retrieval_ms: float | None = None  # CPU time per query, single-query mode (no LLM)
    llm_latency_s: float | None = None  # per query, N samples assumed in parallel
    llm_tokens: float | None = None  # prompt + completion tokens per query
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)


class MissingGenerations(RuntimeError):
    pass


class Experiment:
    def __init__(self, config_path: Path, threads: int = 8):
        self.config_path = Path(config_path)
        self.cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.name = self.cfg.get("name", self.config_path.stem)
        self.ds = load_beir(self.cfg["dataset"], split=self.cfg.get("split", "test"))
        self.depth = int(self.cfg.get("depth", 1000))
        self.metrics = tuple(self.cfg.get("metrics", ["ndcg@10", "recall@10", "recall@100", "mrr@10"]))
        self.qids = sorted(self.ds.queries)
        self.threads = threads
        self.results: dict[str, SystemResult] = {}
        self._bm25: BM25Retriever | None = None
        self._encoders: dict[str, Encoder] = {}
        self._indexes: dict[str, DenseIndex] = {}
        self._qvecs: dict[str, np.ndarray] = {}
        self._reranker = None
        self.out_dir = paths.results_dir() / self.name

    # ------------------------------------------------------------------ components
    def bm25(self) -> BM25Retriever:
        if self._bm25 is None:
            ids = list(self.ds.corpus)
            self._bm25 = BM25Retriever()
            self._bm25.index(
                ids, [doc_text_flat(self.ds.corpus[d].title, self.ds.corpus[d].text) for d in ids]
            )
        return self._bm25

    def encoder(self, name: str) -> Encoder:
        if name not in self._encoders:
            self._encoders[name] = Encoder(ENCODERS[name], threads=self.threads)
        return self._encoders[name]

    def index(self, name: str) -> DenseIndex:
        if name not in self._indexes:
            idx = load_index(self.ds.name, name)
            if idx is None:
                raise FileNotFoundError(
                    f"No {name} index for {self.ds.name}. Run: "
                    f"uv run hyde-rag encode --dataset {self.ds.name} --encoders {name}"
                )
            self._indexes[name] = idx
        return self._indexes[name]

    def query_vecs(self, enc: str) -> np.ndarray:
        if enc not in self._qvecs:
            path = index_path(self.ds.name, enc) / f"queries_{self.ds.split}.npz"
            if path.exists():
                data = np.load(path)
                if list(data["ids"]) == self.qids:
                    self._qvecs[enc] = data["vecs"]
            if enc not in self._qvecs:
                vecs = self.encoder(enc).encode([self.ds.queries[q] for q in self.qids], is_query=True)
                np.savez(path, ids=np.array(self.qids), vecs=vecs)
                self._qvecs[enc] = vecs
        return self._qvecs[enc]

    def _dense_run(self, enc: str, vecs: np.ndarray, qids: list[str] | None = None) -> Run:
        qids = qids or self.qids
        hits = self.index(enc).search(vecs, k=self.depth)
        return {q: dict(h) for q, h in zip(qids, hits, strict=True)}

    def generations(self, kind: str, instruction: str, generator: str) -> dict[str, list[str]]:
        """Cached LLM outputs per query, ordered by sample index."""
        path = cache_path(self.ds.name, kind, instruction, generator)
        by_q: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for rec in JsonlCache(path).records():
            by_q[rec.item_id].append((rec.sample, rec.text))
        return {q: [t for _, t in sorted(v)] for q, v in by_q.items()}

    def _llm_stats(self, kind: str, instruction: str, generator: str, n: int) -> tuple[float, float]:
        path = cache_path(self.ds.name, kind, instruction, generator)
        by_q: dict[str, list] = defaultdict(list)
        for rec in JsonlCache(path).records():
            if rec.sample < n:
                by_q[rec.item_id].append(rec)
        lat = [max(r.latency_s for r in recs) for recs in by_q.values()]  # N calls in parallel
        tok = [sum(r.prompt_tokens + r.completion_tokens for r in recs) for recs in by_q.values()]
        return float(np.mean(lat)), float(np.mean(tok))

    def hyde_passages(self, spec: dict) -> dict[str, list[str]]:
        n = int(spec["n"])
        gens = self.generations("hyde", spec["instruction"], spec["generator"])
        short = [q for q in self.qids if len(gens.get(q, [])) < n]
        if short:
            raise MissingGenerations(
                f"{spec['name']}: {len(short)}/{len(self.qids)} queries have fewer than {n} cached passages "
                f"from {spec['generator']}. Generate them with: uv run hyde-rag generate --experiment "
                f"{self.config_path} --system {spec['name']}"
            )
        return {q: gens[q][:n] for q in self.qids}

    def passage_vecs(self, enc: str, spec: dict, passages: dict[str, list[str]]) -> dict[str, np.ndarray]:
        """Encode hypothetical passages as documents; cached per (encoder, instruction, generator)."""
        slug = spec["generator"].replace("/", "__")
        path = index_path(self.ds.name, enc) / "hyde" / spec["instruction"] / f"{slug}.npz"
        store: dict[str, np.ndarray] = {}
        if path.exists():
            data = np.load(path, allow_pickle=False)
            store = {k: data[k] for k in data.files}
        need = [
            (q, i, t) for q, ps in passages.items() for i, t in enumerate(ps) if f"{q}\x1f{i}" not in store
        ]
        if need:
            vecs = self.encoder(enc).encode([t for _, _, t in need], is_query=False)
            for (q, i, _), v in zip(need, vecs, strict=True):
                store[f"{q}\x1f{i}"] = v
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **store)
        return {q: np.stack([store[f"{q}\x1f{i}"] for i in range(len(ps))]) for q, ps in passages.items()}

    # ------------------------------------------------------------------ systems
    def run_system(self, spec: dict) -> SystemResult:
        kind = spec["type"]
        res = SystemResult(spec["name"], spec, run={})
        if kind == "bm25":
            res.run = self.bm25().search({q: self.ds.queries[q] for q in self.qids}, k=self.depth)
        elif kind == "bm25_q2d":
            passages = self.hyde_passages(spec)
            expanded = {q: " ".join([self.ds.queries[q]] * 5 + passages[q]) for q in self.qids}
            res.run = self.bm25().search(expanded, k=self.depth)
            res.llm_latency_s, res.llm_tokens = self._llm_stats(
                "hyde", spec["instruction"], spec["generator"], spec["n"]
            )
        elif kind == "dense":
            res.run = self._dense_run(spec["encoder"], self.query_vecs(spec["encoder"]))
        elif kind == "hyde":
            enc = spec["encoder"]
            pvecs = self.passage_vecs(enc, spec, self.hyde_passages(spec))
            qv = self.query_vecs(enc)
            vecs = np.stack([hyde_vector(qv[i], pvecs[q]) for i, q in enumerate(self.qids)])
            res.run = self._dense_run(enc, vecs)
            res.llm_latency_s, res.llm_tokens = self._llm_stats(
                "hyde", spec["instruction"], spec["generator"], spec["n"]
            )
        elif kind == "rrf":
            res.run = rrf_runs([self.results[i].run for i in spec["inputs"]], depth=self.depth)
            inputs = [self.results[i] for i in spec["inputs"]]
            llm = [r for r in inputs if r.llm_latency_s is not None]
            if llm:
                res.llm_latency_s = max(r.llm_latency_s for r in llm)
                res.llm_tokens = sum(r.llm_tokens or 0 for r in llm)
        elif kind == "multi_query":
            res.run = self._multi_query(spec)
            res.llm_latency_s, res.llm_tokens = self._llm_stats(
                "multi_query", "rewrite_k4", spec["generator"], 1
            )
        elif kind == "rerank":
            res.run = self._rerank(self.results[spec["input"]].run, int(spec.get("depth", 100)))
            base = self.results[spec["input"]]
            res.llm_latency_s, res.llm_tokens = base.llm_latency_s, base.llm_tokens
        else:
            raise ValueError(f"Unknown system type: {kind}")
        if self.cfg.get("remove_self_hits"):
            res.run = remove_self_hits(res.run)
        res.retrieval_ms = self.probe_latency(spec)
        return res

    def _multi_query(self, spec: dict) -> Run:
        rewrites = self.generations("multi_query", "rewrite_k4", spec["generator"])
        missing = [q for q in self.qids if q not in rewrites]
        if missing:
            raise MissingGenerations(f"{spec['name']}: {len(missing)} queries lack cached rewrites.")
        variants = {q: [self.ds.queries[q], *parse_rewrites(rewrites[q][0])] for q in self.qids}
        rankings: dict[str, list[list[str]]] = defaultdict(list)
        flat = [(q, j, text) for q, vs in variants.items() for j, text in enumerate(vs)]
        keys = [f"{q}\x1f{j}" for q, j, _ in flat]
        for leg in spec.get("legs", ["bm25"]):
            if leg == "bm25":
                run = self.bm25().search(
                    {k: t for k, (_, _, t) in zip(keys, flat, strict=True)}, k=self.depth
                )
            else:
                enc = leg.split(":", 1)[1]
                vecs = self.encoder(enc).encode([t for _, _, t in flat], is_query=True)
                run = self._dense_run(enc, vecs, qids=keys)
            for k, (q, _, _) in zip(keys, flat, strict=True):
                rankings[q].append(sorted(run[k], key=lambda d, r=run[k]: (-r[d], d)))
        return {q: rrf(rankings[q], depth=self.depth) for q in self.qids}

    def reranker(self):
        if self._reranker is None:
            from hyde_rag.retrieval.rerank import DEFAULT_RERANKER, CrossEncoderReranker, cache_file_for

            model_id = self.cfg.get("reranker", DEFAULT_RERANKER)
            cache = cache_file_for(paths.index_dir() / self.ds.name, model_id)
            self._reranker = CrossEncoderReranker(model_id, cache_file=cache, threads=self.threads)
        return self._reranker

    def _doc_text(self, did: str) -> str:
        d = self.ds.corpus[did]
        return doc_text_dense(d.title, d.text)

    def _rerank(self, run: Run, depth: int) -> Run:
        pairs, owners = [], []
        for q in self.qids:
            top = sorted(run.get(q, {}).items(), key=lambda kv: (-kv[1], kv[0]))[:depth]
            for did, _ in top:
                pairs.append((q, did, self.ds.queries[q], self._doc_text(did)))
                owners.append((q, did))
        scores = self.reranker().score(pairs)
        out: Run = defaultdict(dict)
        for (q, did), s in zip(owners, scores, strict=True):
            out[q][did] = s
        return {q: out.get(q, {}) for q in self.qids}

    # ------------------------------------------------------------------ latency
    def probe_latency(self, spec: dict) -> float:
        """Mean CPU milliseconds per query in single-query mode, caches bypassed (no LLM time)."""
        probe = self.qids[:LATENCY_PROBE_QUERIES]
        kind = spec["type"]
        start = time.perf_counter()
        if kind in ("bm25", "bm25_q2d"):
            for q in probe:
                self.bm25().search({q: self.ds.queries[q]}, k=self.depth)
        elif kind in ("dense", "hyde"):
            enc = self.encoder(spec["encoder"])
            passages = self.hyde_passages(spec) if kind == "hyde" else {}
            for q in probe:
                texts = [self.ds.queries[q]]
                qv = enc.encode(texts, is_query=True)[0]
                if kind == "hyde":
                    qv = hyde_vector(qv, enc.encode(passages[q], is_query=False))
                self.index(spec["encoder"]).search(qv[None, :], k=self.depth)
        elif kind == "rrf":
            return sum(self.results[i].retrieval_ms or 0 for i in spec["inputs"])
        elif kind == "multi_query":
            rewrites = self.generations("multi_query", "rewrite_k4", spec["generator"])
            for q in probe:
                variants = [self.ds.queries[q], *parse_rewrites(rewrites[q][0])]
                rankings = []
                for leg in spec.get("legs", ["bm25"]):
                    if leg == "bm25":
                        for v in variants:
                            r = self.bm25().search({"x": v}, k=self.depth)["x"]
                            rankings.append(sorted(r, key=lambda d, r=r: -r[d]))
                    else:
                        enc = leg.split(":", 1)[1]
                        vecs = self.encoder(enc).encode(variants, is_query=True)
                        for hits in self.index(enc).search(vecs, k=self.depth):
                            rankings.append([d for d, _ in hits])
                rrf(rankings, depth=self.depth)
        elif kind == "rerank":
            model = self.reranker()._model
            run = self.results[spec["input"]].run
            depth = int(spec.get("depth", 100))
            for q in probe[:5]:
                top = sorted(run[q], key=lambda d: -run[q][d])[:depth]
                model.predict([(self.ds.queries[q], self._doc_text(d)) for d in top], batch_size=32)
            elapsed = (time.perf_counter() - start) * 1000 / min(5, len(probe))
            return elapsed + (self.results[spec["input"]].retrieval_ms or 0)
        return (time.perf_counter() - start) * 1000 / len(probe)

    # ------------------------------------------------------------------ orchestration
    def run_all(self, skip_missing: bool = True, log=print) -> None:
        for spec in self.cfg["systems"]:
            name = spec["name"]
            deps = spec.get("inputs", []) + ([spec["input"]] if "input" in spec else [])
            if any(d not in self.results for d in deps):
                log(f"- {name}: skipped (needs {', '.join(d for d in deps if d not in self.results)})")
                continue
            try:
                t0 = time.perf_counter()
                res = self.run_system(spec)
            except MissingGenerations as e:
                if not skip_missing:
                    raise
                log(f"- {name}: skipped ({e})")
                continue
            res.metrics = evaluate(self.ds.qrels, res.run, self.metrics)
            self.results[name] = res
            ndcg = np.mean(list(res.metrics["ndcg@10"].values()))
            log(f"- {name}: nDCG@10={ndcg:.4f}  ({time.perf_counter() - t0:.0f}s)")

    def report(self) -> dict:
        """Write runs (top-100), per-system metrics with CIs and significance, and a table."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "runs").mkdir(exist_ok=True)
        baseline_for = self.cfg.get("baselines", {})
        summary: dict = {
            "experiment": self.name,
            "dataset": self.ds.name,
            "split": self.ds.split,
            "n_queries": len(self.qids),
            "systems": {},
        }
        pvals: dict[str, dict[str, float]] = defaultdict(dict)
        for name, res in self.results.items():
            top = {
                q: dict(sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:SAVED_RUN_DEPTH])
                for q, d in res.run.items()
            }
            with gzip.open(self.out_dir / "runs" / f"{name}.json.gz", "wt", encoding="utf-8") as f:
                json.dump(top, f)
            entry: dict = {
                "spec": res.spec,
                "retrieval_ms": res.retrieval_ms,
                "llm_latency_s": res.llm_latency_s,
                "llm_tokens_per_query": res.llm_tokens,
                "metrics": {},
            }
            baseline = res.spec.get("baseline") or baseline_for.get(
                res.spec.get("encoder", ""), self.cfg.get("baseline")
            )
            for metric, per_q in res.metrics.items():
                mean_, lo, hi = bootstrap_ci(per_q)
                m = {"mean": mean_, "ci95": [lo, hi]}
                if baseline and baseline in self.results and baseline != name:
                    cmp_ = compare_paired(per_q, self.results[baseline].metrics[metric])
                    m.update(
                        {
                            "baseline": baseline,
                            "delta": cmp_.delta,
                            "delta_ci95": [cmp_.ci_low, cmp_.ci_high],
                            "p": cmp_.p_value,
                        }
                    )
                    pvals[metric][name] = cmp_.p_value
                entry["metrics"][metric] = m
            summary["systems"][name] = entry
        for metric, ps in pvals.items():
            for name, p_adj in holm(ps).items():
                summary["systems"][name]["metrics"][metric]["p_holm"] = p_adj
        (self.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
