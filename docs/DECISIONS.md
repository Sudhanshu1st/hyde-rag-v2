# Decisions log

Each entry records what was decided, why, and the evidence. "Paper" means Gao et al., ACL 2023
(arXiv:2212.10496). Entries marked **(pending)** need sign-off at a checkpoint.

## Environment and layout

**D-ENV-1 Python and pins.** `requires-python >=3.11`; development on CPython 3.12.14 via `uv`.
Direct dependencies are pinned with `==` in `pyproject.toml`, and `uv.lock` pins transitives.
numpy is held at 2.4.6 and scipy at 1.17.1 because numpy 2.5 and scipy 1.18 dropped Python 3.11
(PyPI metadata, checked 2026-10-14). The pinned set resolves and installs on Windows 11
(111 packages, `uv sync --all-extras`).

**D-ENV-2 CPU-only PyTorch.** A `pytorch-cpu` index (`https://download.pytorch.org/whl/cpu`) is
pinned for `torch` so Linux CI doesn't download CUDA wheels.

**D-ENV-3 No `make` on Windows.** `make` isn't installed on the dev machine. Every Make target is
a thin wrapper around a cross-platform CLI (`uv run hyde-rag <command>`), so Windows users can
run the CLI directly or install GNU make (e.g. `choco install make` / `scoop install make`).

**D-ENV-4 TLS for downloads.** On Windows, Python verifies TLS against the OS store, which
doesn't lazily install newer roots. The BEIR mirror (UKP, HARICA/GEANT chain) therefore fails
verification even though the chain is valid (`openssl s_client`: verify return code 0). Download
scripts verify against `certifi`'s bundle. Verification is never disabled.

**D-ENV-5 Layout.** As proposed in the brief, with three additions:
- `src/hyde_rag/cli.py`: the task runner (see D-ENV-3).
- `configs/experiments/`: one YAML file per experiment.
- `notebooks/`: a self-contained Kaggle/Colab notebook for GPU corpus encoding (see D-COMP-1).

## Retrieval

**D-RET-1 BM25 = `bm25s`, configured to match Pyserini "BM25 flat".** The paper's BM25 row
matches Pyserini BM25-*flat* on every dataset (SciFact .6789/.9253; multifield is .6647), per
Pyserini 2CR. Pyserini flat uses:
- k1=0.9, b=0.4;
- Lucene `DefaultEnglishAnalyzer` (StandardTokenizer, possessive filter, lowercase, Lucene's
  33-word English stopword list, Porter stemmer);
- `contents = title + "\n" + text`;
- `--remove-query`.

We use `bm25s` (no Java) with `method="lucene"`, k1=0.9, b=0.4, PyStemmer `"porter"` (the
original Porter algorithm, as in Lucene's PorterStemFilter, not Snowball "english"), and
Lucene's stopword list. Known residual differences:
- bm25s's regex tokenizer drops 1-character tokens, while StandardTokenizer keeps them;
- there's no possessive filter;
- Lucene stores document length lossily (1-byte norms), while bm25s uses exact lengths.

The size of the gap is measured at the sanity gate and reported, not assumed.

**D-RET-2 Contriever (paper parity).** `facebook/contriever` via `transformers`, with:
- mean pooling over the attention mask;
- `max_length=512`, no L2 normalisation, inner product;
- documents as `title + " " + text` (Contriever repo `beir_utils.py`; Pyserini's prebuilt
  index uses the same p_max_len 512).

Target: SciFact nDCG@10 .6493 (Pyserini), paper 64.9.

**D-RET-3 Modern embedder.** `BAAI/bge-base-en-v1.5` (MIT): CLS pooling and normalisation from
its sentence-transformers config, max length 512. Queries are prefixed with `"Represent this
sentence for searching relevant passages: "`. The model card says passages get no
instruction, and the ST config defines no prompts, so we pass the prefix explicitly.
Hypothetical documents are encoded as passages, without the prefix.

**D-RET-4 Exact search, no FAISS.** The largest corpus is TREC-COVID: 171,332 × 768 vectors,
about 263 MB in float16. Exact inner-product search in numpy is fast enough and matches the
paper's flat (exact) FAISS index, so `faiss-cpu` isn't a dependency.

**D-RET-5 Hybrid.** Reciprocal Rank Fusion (Cormack et al., 2009) with k=60 over each leg's
top-1000. In Hybrid+HyDE, the HyDE vector drives the dense leg. The optional Query2doc-style
BM25 expansion uses `query × 5 + hypothetical documents` (Wang et al., 2023).

**D-RET-6 Multi-query (RAG-Fusion).** One LLM call returns K=4 rewrites. The original query
plus the 4 rewrites are each retrieved and fused with RRF (k=60). One call per query keeps its
LLM cost comparable to HyDE.

**D-RET-7 Reranker.**
- Default model: `cross-encoder/ms-marco-MiniLM-L6-v2`. This is the canonical ID; the old
  `...MiniLM-L-6-v2` redirects to it. Apache-2.0, 22.7M parameters.
- Depth: top-100 of each system.
- Scores are cached per (query_id, doc_id), so overlapping candidate lists across systems are
  scored once.
- `BAAI/bge-reranker-v2-m3` (568M parameters) is GPU-only in practice (see D-COMP-1).

## HyDE

**D-HYDE-1 Query vector = paper Eq. 8.** v = (1/(N+1)) · [Σ_k f(d̂_k) + f(q)]. The query's own
embedding is included, as the paper's Eq. 8 and the official repo's `hyde.py` `encode()` do,
with no normalisation. Each generation is whitespace-stripped before encoding, as in the
official notebook.

**D-HYDE-2 N.** The paper text gives no N. The official repo's generator defaults to `n=8`, and
its DL19 notebook uses n=8 ("8rep" output). We generate 8 samples once and evaluate
N ∈ {1, 4, 8} on nested prefixes (samples 0..N-1). Nesting makes the N comparison paired and
avoids re-sampling noise.

**D-HYDE-3 Sampling deviation.** The paper used text-davinci-003 at temperature 0.7, top_p 1,
max_tokens 512, and stop `"\n\n\n"`.
- Claude Opus 5.5 rejects sampling parameters. Claude Haiku 5.5 accepts only temperature 1
  (its default).
- So the N samples are drawn as independent calls at default sampling.
- We keep max_tokens=512 for visible text and the `"\n\n\n"` stop sequence.

**D-HYDE-4 Thinking and effort.** Bulk generation uses `effort=low`.
- Haiku 5.5: thinking `disabled`, which is allowed at effort ≤ high. This is closest to the
  paper's non-reasoning generator.
- Opus 5.5: thinking can't be disabled, so it runs adaptive at effort low with
  `max_tokens` raised to leave room for thinking.
- Thinking tokens are billed as output and are recorded in the cost per query.

**D-HYDE-5 Instructions.** Paper-parity runs use the ACL camera-ready Appendix A.1 wording
verbatim (`configs/hyde_instructions.yaml`). It differs slightly from arXiv v1 and from the
repo's `promptor.py`, e.g. "support or refute" vs "support/refute" and a trailing period.

**D-HYDE-6 Passage-only system prompt.** Chat models tend to prepend "Here is a passage…",
which would leak into the embedding. A one-line system prompt asks for the passage only. This
is a deviation from the paper's raw completion prompt.

## LLM layer

**D-LLM-1 Provider-agnostic client.** `LLMClient` has two backends:
- Anthropic, the default, via the official `anthropic` SDK;
- an OpenAI-compatible backend, via the `openai` SDK, for open-weights models on Groq or
  Ollama ("private deployment" mode).

Claude is never called through an OpenAI-compatible shim.

**D-LLM-2 Bulk generation.**
- Bulk runs use the Message Batches API (50% price).
- Latency is measured separately with interactive calls on a 30-query sample per system,
  because batch latency is meaningless.
- Reported LLM cost per query uses standard (non-batch) prices, i.e. what an online system
  pays, with the batch price in a separate column.

**D-LLM-3 Cache.** JSONL files, one per (dataset, generator, instruction). The key is the
sha256 of backend, model, the canonical JSON of request parameters, and the sample index. Each
record stores `query_id`, text, usage, stop_reason, model and timestamp. **No query or corpus
text is stored**, so the committed `cache/beir/` files don't redistribute dataset text.

**D-LLM-4 Cost guard.** Before any bulk run:
- input tokens are counted exactly with `messages.count_tokens`;
- output tokens are estimated from a 20-call pilot, which also measures thinking.

A run estimated above $5 (`HYDE_RAG_COST_LIMIT_USD`) stops unless re-run with `--approve-cost`.
Planning totals are in `docs/COST_PLAN.md`, generated by `scripts/estimate_cost.py`.

**D-LLM-5 Model roles (pending).** Defaults in `configs/run_plan.yaml`:
- HyDE/multi-query bulk = `claude-haiku-5-5`
- top-generator ablation = `claude-opus-5-5`
- answerer = `claude-sonnet-5-5`
- judge = `claude-opus-5-5`
- food eval-question generator = `claude-opus-5-5`, which deliberately differs from the HyDE
  generator
- synthetic fine-tuning queries = `claude-haiku-5-5`, which deliberately differs from the
  question generator

## Evaluation

**D-EVAL-1 Metrics with trec_eval semantics.** We use `pytrec-eval-terrier`, the maintained fork
with Windows wheels; plain `pytrec_eval` is sdist-only.
- nDCG@10 = `ndcg_cut_10`; Recall@10/100 = `recall_10/100` (relevance ≥ 1).
- MRR@10 = `recip_rank` on runs truncated to 10.
- We chose this over ranx for exact parity with Pyserini's numbers. ranx also lacks a paired
  bootstrap and has an open pandas-3 bug.

**D-EVAL-2 Capped recall for TREC-COVID.** The paper's TREC-COVID R@100 values are BEIR's
*capped* R_cap@100 (BM25 49.8 = BEIR Table 9), while Pyserini reports uncapped recall (.1091).
We report both and compare with the paper on the capped version. TREC-COVID qrels contain two
`-1` judgements, which are treated as non-relevant.

**D-EVAL-3 ArguAna self-hits.** As with Pyserini's `--remove-query` and BEIR's
`ignore_identical_ids`, a document whose id equals the query id is removed from the run.

**D-EVAL-4 Uncertainty.**
- 95% CIs come from a paired bootstrap over queries (10,000 resamples, fixed seed).
- Significance against the naive-dense baseline of the same encoder uses a paired
  randomisation test (10,000 permutations).
- Holm–Bonferroni correction is applied across the systems compared on a dataset.

**D-EVAL-5 Judge = custom claim-level judge, not RAGAS.** RAGAS 0.4.3 hard-depends on LangChain
(imported at module load) and hasn't been released since 2026-01. Our judge:
- splits each answer into atomic claims;
- judges support against the cited chunks using structured output (pydantic);
- is validated against 50 hand labels before its numbers are trusted.

## Compute

**D-COMP-1 Measured CPU throughput on the dev laptop.** Intel i5-1235U, 16 GB RAM, no GPU, on
AC power. Measured on 256 SciFact abstracts (fp32, batch 32):

| Model | Throughput |
|---|---|
| Contriever | 0.9 docs/s |
| bge-base | 2.0 docs/s |
| MiniLM-L6 cross-encoder | 14 pairs/s |

Consequences:
- SciFact can be encoded locally: about 1.6 h for Contriever and 45 min for bge.
- TREC-COVID (171K docs) would take about 53 h and 24 h, so it is encoded on a free GPU
  (Kaggle/Colab) with a self-contained notebook, and the embeddings are downloaded back.
- The same numbers make HyDE's query-side cost visible: encoding 8 passages on CPU takes
  seconds per query. This is reported in the latency columns.

## Licensing

**D-LIC-1** Code: MIT. Dataset and model licences are in `NOTICE`.
`facebook/contriever` is CC BY-NC 4.0 (repo LICENSE; the HF card has no licence field). It is
used only for paper-parity benchmark runs, and the API/demo default to `bge-base-en-v1.5` (MIT).

**D-LIC-2** PDF parsing uses `pdfplumber` (MIT), which gives page numbers plus font size and
name for heading detection. PyMuPDF is avoided because it is AGPL-3.0.

## Food corpus (public data only)

**D-DATA-1 Composition (draft, 59 docs).** `data_sources.yaml` lists the documents, with
licences verified against each publisher's own policy page on 2026-10-14:
- reference codes: FDA Food Code 2026, Codex CXC 1-1969, FSSAI Schedule 4 + 2026 amendment,
  FSSAI recall rules;
- FDA preventive-controls guidance chapters, Listeria, sanitation, seafood/juice HACCP, and
  recall guidance and manuals;
- FSIS compliance guidelines and directives;
- 12 FDA warning letters, for corrective-action language;
- 4 openFDA recall snapshots, for root-cause questions.

51 are committable (US public domain or CC0) and 8 are download-only.

**D-DATA-2 "Public domain" is checked per document.** The text of each candidate PDF was
scanned for permission notices. Two flagship US documents reproduce third-party material
"used/reprinted with permission":
- FDA Food Code 2026, Annex 4;
- FSIS Revised Appendix A, p.29.

Both are download-only, as is PCHF Appendix 2 (forms "adapted from" FSPCA), as a precaution.
Codex (CC BY-NC-SA 3.0 IGO) and FSSAI (reuse with acknowledgement, not US public domain) are
download-only under the project rule.

**D-DATA-3 Warning letters (pending).** Letters are public records but name individuals
(recipients, signatories) and carry FOIA (b)(4)/(b)(6)/(b)(7)(C) redactions. Proposal: store
only the extracted letter body with personal names masked, keeping firm names, and never the
site chrome.

**D-DATA-4 Polite, reproducible downloading.**
- fda.gov: `Crawl-Delay: 30` plus bot detection, so downloads are single-threaded with ≥30 s
  between requests and back-off on errors.
- usda.gov/fsis.usda.gov: these return 403 to scripted clients from the dev network, so the
  script falls back to "download manually to this path" and verifies sha256. These files are
  public domain, so once committed nobody needs to re-download them.
- openFDA: snapshots are stored with their query string and `meta.last_updated`.
- We never try to evade bot detection.

**D-DATA-5 FSSAI Schedule 4 is amended.** The 2021 compendium's Schedule 4 is partly
superseded by Amendment No. 5 (23 June 2026), so both are included and eval questions cite the
amended text where they differ.
