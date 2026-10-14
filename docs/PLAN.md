# Plan

Research question: **when does HyDE help compared with standard retrieval, and does that still
hold for food-safety quality documents?** Each phase ends at a checkpoint.

## Experiment matrix

### Systems
Each system is one config entry.

| # | System | Variants |
|---|---|---|
| S1 | BM25 (bm25s, Pyserini-flat settings) | — |
| S2 | Naive dense | Contriever (parity) · bge-base-en-v1.5 |
| S3 | HyDE (Eq. 8) | × encoder {Contriever, bge} × generator {gpt-oss-120b, gpt-oss-20b, Qwen3} (Groq free tier) × N {1, 4, 8} |
| S4 | Hybrid BM25 + dense (RRF) | × encoder |
| S5 | Hybrid + HyDE (HyDE drives the dense leg) | ± Query2doc BM25 expansion |
| S6 | Multi-query / RAG-Fusion (K=4 rewrites, RRF) | (gpt-oss-120b) |
| S7 | + cross-encoder rerank (top-100) | applied to S1–S6 |

### Datasets

| Dataset | Docs | Test queries | Use | Notes |
|---|---:|---:|---|---|
| SciFact | 5,183 | 300 (+809 train) | primary: retrieval + generation | train split = dev for τ calibration |
| TREC-COVID | 171,332 | 50 | secondary: retrieval | dense encoding on free GPU |
| ArguAna | 8,674 | 1,406 | optional: retrieval only | dropped from the free-tier plan (1,406 long queries) |
| Food-quality (ours) | 43 public docs | ~250 (30% dev / 70% test) | domain adaptation | human-reviewed |

## Phases

1. **Plan and verify (this checkpoint).** Libraries, model IDs, dataset loaders, food-source
   licences and the cost plan.
2. **BEIR retrieval.**
   - Sanity gate first: BM25 and Contriever on SciFact must land within about 2 nDCG@10 points
     of the paper (67.9 / 64.9). If not, stop and document.
   - Then HyDE (N sweep, generator ablation), bge, hybrid, multi-query and rerank on SciFact,
     then TREC-COVID, then optionally ArguAna.
   - Report CIs, significance, latency and cost per query.
3. **Generation.**
   - Citations (`doc_id:page:chunk`) with post-hoc validation.
   - Two-layer abstention (reranker gate τ calibrated on dev, plus a prompt rule).
   - Claim-level faithfulness judge, validated on 50 hand labels.
   - SciFact 100-claim sample; the original SciFact release has SUPPORT/CONTRADICT labels for
     these claims, usable for answer correctness (to be verified).
   - Unanswerable SciFact cases are built by removing gold evidence from the index (proposal).
4. **Food corpus and eval set.**
   - `data_sources.yaml` with licences and sha256 values.
   - Structure-aware chunking (pdfplumber headings) and a draft question set.
   - Review CSV, then **stop for your review**.
5. **Domain adaptation.**
   - F1: domain HyDE instruction.
   - F2: bge fine-tune on synthetic queries with hard negatives.
   - F3: optional reranker fine-tune.
   - Compare base / F1 / F2 / F1+F2, each ± rerank; tune on dev, report on test.
6. **Packaging.**
   - FastAPI `/query`, Streamlit demo, Dockerfile, CI (lint, tests, cached-generation smoke
     eval, no keys).
   - README with results generated from `results/`, `RESULTS.md`, and a blog draft.
   - Checkpoint before any GitHub push.

## Budget: zero spend

Nothing costs money. Every LLM call uses Groq's free plan, or any other OpenAI-compatible
endpoint such as local Ollama. Encoders and rerankers run locally. The constraint is free
quota: 200K tokens and 1K requests per model per day.

`docs/QUOTA_PLAN.md` (generated) estimates about 5.7M tokens in total, which is about 15 days
of unattended background generation if the phases run back to back. Phases overlap where
their models differ. The runner resumes across days automatically, and every generation is
cached and committed, so reproducing the results needs no key and no wait.
