# HyDE RAG v2

**When does HyDE (Hypothetical Document Embeddings) actually help, and does it still hold for
food-safety quality documents?** An open-source RAG system and benchmark, built and run entirely
on free tiers.

> **Status: work in progress.** Retrieval, generation and evaluation code is in place, and the
> BEIR experiments are running. Final results tables, the findings write-up, the API, the demo
> and CI are still to come. Numbers will only appear here once they are generated from `results/`.

## What's here

- **Retrieval, every method a config entry:**
  - BM25, configured to match Pyserini's "flat" baseline;
  - dense retrieval with Contriever (paper parity) and bge-base-en-v1.5;
  - HyDE following the paper's Eq. 8;
  - hybrid search with Reciprocal Rank Fusion (RRF), and Query2doc-style expansion;
  - multi-query (RAG-Fusion);
  - cross-encoder reranking.
- **Paper reproduction** of Gao et al. (2023) on BEIR SciFact and TREC-COVID, with paired
  bootstrap confidence intervals and significance tests.
- **Grounded generation:**
  - inline citations (`doc_id:page:chunk`), validated after generation;
  - a two-layer "not found" fallback: a reranker-score gate plus a prompt rule;
  - a claim-level faithfulness judge.
- **A food-safety corpus** of 41 public documents (FDA, USDA FSIS, Codex, FSSAI), with
  per-document licence checks in [`data_sources.yaml`](data_sources.yaml) and a human-reviewed
  question set.
- **Zero cost:**
  - LLM calls use Groq's free tier, or any OpenAI-compatible endpoint, e.g. local Ollama;
  - a guard refuses API keys that are on a paid plan;
  - generations are cached and committed, so results reproduce without a key.

## Quick start (Windows, macOS, Linux)

```
uv sync --all-extras
uv run python scripts/download_beir.py scifact
uv run hyde-rag encode --dataset scifact --encoders contriever bge-base
uv run hyde-rag run configs/experiments/scifact.yaml
```

LLM-based systems read their outputs from `cache/`. To generate new ones, put a free Groq key
in `.env` (see `.env.example`) and run `uv run hyde-rag generate --experiment <config>`.

## Docs

- [`docs/PLAN.md`](docs/PLAN.md): the experiment plan
- [`docs/DECISIONS.md`](docs/DECISIONS.md): every design decision, with its evidence
- [`docs/QUOTA_PLAN.md`](docs/QUOTA_PLAN.md): the free-tier budget
- [`NOTICE`](NOTICE): dataset and model licences

## Credits

Gao, Ma, Lin, Callan. *Precise Zero-Shot Dense Retrieval without Relevance Labels.* ACL 2023.
[arXiv:2212.10496](https://arxiv.org/abs/2212.10496). This repository re-implements the method
from the paper. Version 1 of this project: <https://github.com/Sudhanshu1st/HyDe_RAG>.

Code: MIT. Data and models keep their own licences; see `NOTICE`.
