"""Cross-platform task runner (D-ENV-3): `uv run hyde-rag <command>`.

Make targets are thin wrappers around these commands, so everything also works on
Windows without `make`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Tuned on the dev laptop (i5-1235U): 8 threads + length-sorted batches of 16 was fastest
# (docs/DECISIONS.md D-COMP-1). Override with --threads.
DEFAULT_THREADS = 8


def cmd_encode(args: argparse.Namespace) -> int:
    from hyde_rag.ingest.beir import load_beir
    from hyde_rag.retrieval.dense import ENCODERS, Encoder, doc_text_dense, encode_corpus, index_path

    ds = load_beir(args.dataset, split=args.split)
    doc_ids = list(ds.corpus)
    texts = [doc_text_dense(ds.corpus[d].title, ds.corpus[d].text) for d in doc_ids]
    for name in args.encoders:
        out = index_path(args.dataset, name)
        if (out / "embeddings.npy").exists() and not args.force:
            print(f"[{name}] already encoded: {out}")
            continue
        encoder = Encoder(ENCODERS[name], threads=args.threads)
        start = time.perf_counter()
        encode_corpus(encoder, doc_ids, texts, out, batch_size=args.batch_size)
        elapsed = time.perf_counter() - start
        print(f"[{name}] {len(texts):,} docs in {elapsed / 60:.1f} min ({len(texts) / elapsed:.2f} docs/s)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from hyde_rag.eval.experiment import Experiment
    from hyde_rag.eval.report import render_table

    exp = Experiment(args.experiment, threads=args.threads)
    print(f"Experiment {exp.name}: {len(exp.qids)} queries, {len(exp.cfg['systems'])} systems")
    exp.run_all()
    exp.report()
    table = render_table(exp.out_dir / "metrics.json")
    (exp.out_dir / "table.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    return 0


def _llm_jobs(exp_cfg: dict, only: set[str] | None) -> dict[tuple[str, str, str], int]:
    """(kind, instruction, generator) -> samples needed, from the experiment's LLM systems."""
    jobs: dict[tuple[str, str, str], int] = {}
    for spec in exp_cfg["systems"]:
        if only and spec["name"] not in only:
            continue
        if spec["type"] in ("hyde", "bm25_q2d"):
            key = ("hyde", spec["instruction"], spec["generator"])
            jobs[key] = max(jobs.get(key, 0), int(spec["n"]))
        elif spec["type"] == "multi_query":
            jobs[("multi_query", "rewrite_k4", spec["generator"])] = 1
    return jobs


def cmd_generate(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor

    import yaml

    from hyde_rag.ingest.beir import load_beir
    from hyde_rag.llm.cache import JsonlCache
    from hyde_rag.llm.client import OpenAICompatClient
    from hyde_rag.llm.quota import Limits, load_limits
    from hyde_rag.llm.runner import PILOT_CALLS, run_bulk
    from hyde_rag.retrieval.hyde import cache_path, hyde_items, load_instruction, multi_query_items

    limits = load_limits()
    unlimited = Limits(rpm=10**6, rpd=10**9, tpm=10**12, tpd=10**15)  # e.g. local Ollama
    by_model: dict[str, list] = {}
    # Experiments are queued in the order given; each model works through its queue alone,
    # so one process uses each free-tier quota exactly once.
    for experiment in args.experiment:
        cfg = yaml.safe_load(Path(experiment).read_text(encoding="utf-8"))
        ds = load_beir(cfg["dataset"], split=cfg.get("split", "test"))
        queries = {q: ds.queries[q] for q in sorted(ds.queries)}
        jobs = _llm_jobs(cfg, set(args.system) if args.system else None)
        if args.generator:
            jobs = {k: v for k, v in jobs.items() if k[2] in args.generator}
        # Sort HyDE before multi-query, so the headline results arrive first.
        for (kind, instruction, generator), n in sorted(jobs.items()):
            if kind == "hyde":
                n = min(n, args.max_samples) if args.max_samples else n
                items = hyde_items(queries, load_instruction(instruction), generator, n)
            else:
                items = multi_query_items(queries, generator)
            path = cache_path(ds.name, kind, instruction, generator)
            by_model.setdefault(generator, []).append((path, items))
            rel = path.relative_to(path.parents[4])
            print(f"- {kind:11s} {instruction:18s} {generator:22s} {len(items):5d} calls -> {rel}")
    if not by_model:
        print("No LLM systems selected.")
        return 0
    client = OpenAICompatClient()

    def run_model(model: str) -> None:
        from hyde_rag.llm.quota import QuotaGuardError, estimate_run

        lim = limits.get(model, unlimited)
        for path, items in by_model[model]:
            try:
                report = run_bulk(
                    client,
                    JsonlCache(path),
                    items,
                    lim,
                    approved=args.yes,
                    limit=PILOT_CALLS if args.pilot_only else None,
                    log=lambda msg: print(msg, flush=True),
                )
            except QuotaGuardError as e:
                print(f"[{model}] stopped before the full run: {e}", flush=True)
                continue
            if args.pilot_only and report.completed:
                est = estimate_run(lim, model, report.remaining, report.mean_tokens)
                print(
                    f"[{model}] {path.parent.parent.name}/{path.parent.name}: measured "
                    f"{report.mean_tokens:.0f} tokens/call; remaining {est.requests:,} calls = "
                    f"{est.tokens:,} tokens = {est.days:.1f} days of free quota",
                    flush=True,
                )

    # Each free-tier model has its own quota, so models run in parallel.
    with ThreadPoolExecutor(max_workers=len(by_model)) as pool:
        for f in [pool.submit(run_model, m) for m in by_model]:
            f.result()
    return 0


def cmd_gen_eval(args: argparse.Namespace) -> int:
    from hyde_rag.eval.generation_eval import GenerationEval

    ev = GenerationEval(args.config)
    summary = ev.run(llm=not args.no_llm, approved=args.yes, log=lambda m: print(m, flush=True))
    print(json.dumps(summary["systems"], indent=2))
    return 0


def cmd_ingest_food(args: argparse.Namespace) -> int:
    from hyde_rag.ingest.food import ingest

    r = ingest(log=lambda m: print(m, flush=True))
    print(
        f"parsed {len(r.parsed)}, unchanged {len(r.reused)}, not downloaded {len(r.missing)}, "
        f"{r.n_chunks:,} chunks total"
    )
    if r.missing:
        print("not downloaded:", ", ".join(r.missing))
    if r.low_text:
        print("WARNING, little extractable text (scanned PDF?):", ", ".join(r.low_text))
    return 0


def cmd_gen_questions(args: argparse.Namespace) -> int:
    from hyde_rag.eval.question_gen import generate

    review = generate(
        llm=not args.no_llm,
        approved=args.yes,
        pilot_only=args.pilot_only,
        batch=args.batch,
        log=lambda m: print(m, flush=True),
    )
    print(f"Review file: {review}")
    return 0


def cmd_build_evalset(args: argparse.Namespace) -> int:
    from hyde_rag.eval.evalset import build

    r = build()
    print(f"kept {r.kept}, rejected {r.rejected}, not reviewed {r.unreviewed}")
    for split, counts in r.by_split.items():
        print(f"  {split}: {sum(counts.values())} items {counts}")
    if r.excluded:
        print(f"excluded (answerable but evidence not found): {', '.join(r.excluded)}")
    return 0


def cmd_quota(args: argparse.Namespace) -> int:
    import runpy

    script = Path(__file__).resolve().parents[2] / "scripts" / "estimate_quota.py"
    sys.argv = [str(script)]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="hyde-rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("encode", help="encode a BEIR corpus with one or more dense encoders")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--encoders", nargs="+", default=["contriever", "bge-base"])
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_encode)

    p = sub.add_parser("run", help="run an experiment config and write results/<name>/")
    p.add_argument("experiment")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("generate", help="generate missing LLM outputs for an experiment (free tier)")
    p.add_argument("--experiment", action="append", required=True, help="repeat to queue several")
    p.add_argument("--max-samples", type=int, help="cap HyDE samples per query (core profile: 4)")
    p.add_argument("--system", action="append", help="limit to these system names")
    p.add_argument("--generator", action="append", help="limit to these generator models")
    p.add_argument("--yes", action="store_true", help="allow runs longer than HYDE_RAG_MAX_QUOTA_DAYS")
    p.add_argument(
        "--pilot-only", action="store_true", help="run 20 calls per job, report measured cost, stop"
    )
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("gen-eval", help="generation eval: verdicts, citations, abstention, faithfulness")
    p.add_argument("config")
    p.add_argument("--no-llm", action="store_true", help="score cached outputs only (no API calls)")
    p.add_argument("--yes", action="store_true", help="allow runs longer than HYDE_RAG_MAX_QUOTA_DAYS")
    p.set_defaults(func=cmd_gen_eval)

    p = sub.add_parser("ingest-food", help="parse downloaded food PDFs into canonical text and chunks")
    p.set_defaults(func=cmd_ingest_food)

    p = sub.add_parser("gen-questions", help="draft food eval questions and write the review CSV")
    p.add_argument("--no-llm", action="store_true", help="rebuild the CSV from cached outputs only")
    p.add_argument("--pilot-only", action="store_true", help="run 20 calls to check quality and cost")
    p.add_argument("--batch", default="main", help="batch name from configs/question_batches.yaml")
    p.add_argument("--yes", action="store_true", help="allow runs longer than HYDE_RAG_MAX_QUOTA_DAYS")
    p.set_defaults(func=cmd_gen_questions)

    p = sub.add_parser("build-evalset", help="reviewed CSV -> eval_sets/food_quality/{dev,test}.jsonl")
    p.set_defaults(func=cmd_build_evalset)

    p = sub.add_parser("quota", help="free-quota plan for all planned LLM runs")
    p.set_defaults(func=cmd_quota)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
