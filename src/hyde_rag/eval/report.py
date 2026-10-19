"""Render results/<experiment>/metrics.json as a Markdown table (no numbers typed by hand)."""

from __future__ import annotations

import json
from pathlib import Path


def _fmt(x: float | None, digits: int = 1, scale: float = 100.0) -> str:
    return "-" if x is None else f"{x * scale:.{digits}f}"


def render_table(metrics_path: Path) -> str:
    data = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    # Recall columns; BEIR's capped recall is added when computed (TREC-COVID, D-EVAL-2).
    recall = [("recall@10", "R@10"), ("recall@100", "R@100")]
    if any("rcap@100" in s["metrics"] for s in data["systems"].values()):
        recall.append(("rcap@100", "R_cap@100"))
    lines = [
        f"### {data['experiment']} ({data['dataset']} {data['split']}, {data['n_queries']} queries)",
        "",
        "nDCG@10 with 95% bootstrap CI; Δ = paired difference vs the system's naive-dense baseline "
        "(Holm-adjusted randomisation-test p). CPU ms = single-query retrieval on the dev laptop "
        "(no LLM); LLM s and tokens are per query (N samples in parallel).",
        "",
        "| System | nDCG@10 [95% CI] | Δ nDCG@10 (p) | "
        + " | ".join(label for _, label in recall)
        + " | MRR@10 | CPU ms | LLM s | LLM tokens |",
        "|---|---|---|" + "---:|" * (len(recall) + 4),
    ]
    for name, s in data["systems"].items():
        m = s["metrics"]
        nd = m["ndcg@10"]
        ci = f"{_fmt(nd['mean'])} [{_fmt(nd['ci95'][0])}, {_fmt(nd['ci95'][1])}]"
        if "delta" in nd:
            p = nd.get("p_holm", nd["p"])
            stars = "**" if p < 0.01 else "*" if p < 0.05 else ""
            delta = f"{nd['delta'] * 100:+.1f}{stars} (p={p:.3f})"
        else:
            delta = (
                "baseline"
                if any(o["metrics"]["ndcg@10"].get("baseline") == name for o in data["systems"].values())
                else "-"
            )
        llm_s = "-" if s["llm_latency_s"] is None else f"{s['llm_latency_s']:.1f}"
        llm_t = "-" if s["llm_tokens_per_query"] is None else f"{s['llm_tokens_per_query']:,.0f}"
        cpu = "-" if s["retrieval_ms"] is None else f"{s['retrieval_ms']:,.0f}"
        recalls = " | ".join(_fmt(m.get(key, {}).get("mean")) for key, _ in recall)
        lines.append(
            f"| {name} | {ci} | {delta} | {recalls} | {_fmt(m.get('mrr@10', {}).get('mean'))} | "
            f"{cpu} | {llm_s} | {llm_t} |"
        )
    lines += ["", "\\* p < 0.05, \\*\\* p < 0.01 after Holm correction."]
    return "\n".join(lines) + "\n"
