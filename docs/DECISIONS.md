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

**D-HYDE-3 Sampling.** The paper used text-davinci-003 at temperature 0.7, top_p 1,
max_tokens 512, and stop `"\n\n\n"`. Our open-weights generators on Groq accept a temperature,
so we keep **temperature 0.7, top_p 1, stop `"\n\n\n"`** for paper parity. The N samples are
drawn as independent calls with a per-sample seed; the paper drew them with `n=8` in one call.

**D-HYDE-4 Reasoning.** gpt-oss models can't switch reasoning off, so they run at
`reasoning_effort=low`. Their completion budget is the paper's 512 tokens for visible text,
plus headroom for the reasoning trace. Qwen3 runs with thinking off, which is closest to the
paper's non-reasoning generator. Reasoning tokens count against the free quota and are
reported in the token-usage columns.

**D-HYDE-5 Instructions.** Paper-parity runs use the ACL camera-ready Appendix A.1 wording
verbatim (`configs/hyde_instructions.yaml`). It differs slightly from arXiv v1 and from the
repo's `promptor.py`, e.g. "support or refute" vs "support/refute" and a trailing period.

**D-HYDE-6 Passage-only system prompt.** Chat models tend to prepend "Here is a passage…",
which would leak into the embedding. A one-line system prompt asks for the passage only. This
is a deviation from the paper's raw completion prompt.

**D-HYDE-7 Markdown stripped from passages before encoding.** gpt-oss-20b wraps passages in
markdown such as `**Abstract**`. `clean_passage()` removes headings, bold and code ticks, and
collapses whitespace, identically for every generator. The cache keeps the raw text.

**D-HYDE-8 Query dilution in Eq. 8 (finding + ablation).** On SciFact with Contriever, HyDE
got *worse* as N grew: gpt-oss-120b scored 0.674 at N=1 and 0.610 at N=4, below Contriever
alone at 0.649. Diagnosis (scratch analysis, reproducible from the cached vectors):
- **Not a bug.** Every individual sample behaves the same way: N=1 with any of samples 0–3
  scores 0.66–0.67.
- **The passages are weaker queries than the claim.** One gpt-oss-120b passage used on its own
  as a Contriever query scores 0.50–0.53, versus 0.649 for the claim. Under bge, passages beat
  the claim (0.76–0.78 vs 0.740).
- **Eq. 8 weights the query 1/(N+1)**, so at N=4 the claim is only 20% of the vector, and
  passage vectors are ~35% longer (|p| 1.85 vs |q| 1.37, since Contriever is unnormalised).
  Normalising alone recovers little (0.610 → 0.625).

Ablation `query_weight: 0.5`: v = 0.5·f(q) + 0.5·mean_k f(d_k), unnormalised like Eq. 8. It was
found on the SciFact test set, so it was confirmed out-of-sample on TREC-COVID before being
reported. Paired Δ vs Eq. 8 at N=4 (from `results/`):

| Dataset | Encoder | Generator | Δ nDCG@10 [95% CI] | p |
|---|---|---|---|---|
| SciFact | Contriever | gpt-oss-120b | 61.0 → 67.7, +6.7 [+4.6, +8.9] | 0.0001 |
| SciFact | Contriever | Qwen3 | 65.2 → 68.4, +3.2 [+1.6, +4.9] | 0.0002 |
| SciFact | bge | gpt-oss-120b | 78.5 → 77.8, −0.7 [−1.9, +0.5] | 0.24 (no effect) |
| TREC-COVID | Contriever | Qwen3 | 54.1 → 60.9, +6.9 [+3.0, +10.8] | 0.0014 |
| TREC-COVID | Contriever | gpt-oss-120b | 46.9 → 58.9, +12.0 [+7.9, +16.3] | 0.0001 |
| TREC-COVID | bge | gpt-oss-120b | 77.2 → 84.8, +7.6 [+4.1, +11.7] | 0.0001 |

The fixed weight helps when passages are weaker queries than the question (Contriever) and is
neutral when they are stronger (bge on SciFact). On TREC-COVID it also rescues bge: with Eq. 8
at N=4, HyDE-bge loses to bge alone (77.2 vs 78.1), but with w=0.5 it gains +6.7 (p=0.002,
Holm). The TREC-COVID bge rows were added after the weight was fixed, so this is out-of-sample
too (2026-10-19). N=1 needs no fix (Eq. 8 already gives the query 50%): on TREC-COVID HyDE-bge
N=1 scores 83.4, within noise of N=4 at w=0.5 (+1.4 [−0.3, +3.6], p=0.17), at a quarter of the
tokens. Contriever + Qwen3 at w=0.5 (60.9) beats the paper's HyDE (59.3) and BM25 (59.3). (A scratch variant that also L2-normalised the vectors reached 68.6 on SciFact;
it is not part of the pipeline and not reported.) The paper-parity numbers (Eq. 8) remain the
headline; the ablation is reported separately.

## LLM layer

**D-LLM-0 Zero spend (owner decision, 2026-10-14).** The project must not cost anything to
build, run or reproduce. So:
- No paid API is used. The Anthropic backend and Claude models from the original brief are
  dropped.
- Every LLM call goes to Groq's free plan, or to any other OpenAI-compatible endpoint,
  e.g. local Ollama.
- Dollar-cost reporting is replaced by token usage and latency per query.

**D-LLM-1 One provider-agnostic client.** `LLMClient` talks to any OpenAI-compatible endpoint
through the `openai` SDK: Groq by default, Ollama for fully local "private deployment" mode.
The interface is backend-neutral, so other backends can be added without touching the
pipeline.

**D-LLM-2 Free-tier scheduling.** Groq's free plan allows each model 30 requests/min,
1K requests/day, 8K tokens/min and 200K tokens/day (published 2026-10-14). The account's real
limits are read from the `x-ratelimit-*` headers.
- The bulk runner paces itself to these limits and honours `429 retry-after`.
- It writes every result to the cache immediately and resumes across days.
- Different models have separate quotas, so roles are spread across models that run in
  parallel.
- Latency per query comes from the measured duration of each HTTP call (pacing waits are
  excluded), with a system's N samples assumed to run in parallel.
- Observed 2026-10-14: the daily token limit is a **rolling 24-hour window**. After about 390
  gpt-oss-120b passages the API returned 429s with retry-after values of minutes, not
  hours, so generation trickles on continuously rather than waiting for a daily reset.

**D-LLM-3 Cache, committed.** JSONL files, one per (dataset, generator, instruction). The key
is the sha256 of base URL, model, the canonical JSON of request parameters, and the sample
index. Each record stores `query_id`, text, token usage, finish reason, model and timestamp.
- BEIR cache records contain **no query or corpus text**.
- All caches are committed, so anyone can reproduce every reported number without a key.

**D-LLM-4 Quota guard instead of a cost guard.**
- `scripts/estimate_quota.py` converts every planned run into requests, tokens and days of
  free quota (`docs/QUOTA_PLAN.md`).
- Before a bulk run, a 20-call pilot measures real tokens per call.
- Runs longer than `HYDE_RAG_MAX_QUOTA_DAYS` (default 1) need `--yes`.
- **Zero-spend safety:** if a key's rate-limit headers show paid-plan limits
  (> 50K tokens/min), the client refuses to run unless `HYDE_RAG_ALLOW_PAID=1`.

**D-LLM-5 Model roles** (`configs/llm.yaml`). Free Groq models only. The Llama models on
Groq are enterprise-only as of 2026-10-14.

| Role | Model | Why |
|---|---|---|
| Headline HyDE + multi-query | `openai/gpt-oss-120b` | Strongest free model, like the paper's InstructGPT |
| Generator ablation, small | `openai/gpt-oss-20b` | Same family, smaller |
| Generator ablation, other family | `qwen/qwen3.8-27b` | Preview model |
| Answerer | `openai/gpt-oss-20b` | |
| Judge | `qwen/qwen3.8-27b` | Must differ from the answerer |
| Food eval-question generator | `openai/gpt-oss-20b` | Must differ from the headline HyDE generator; moved from Qwen3 on 2026-10-15 because gpt-oss-20b had finished its HyDE share and was idle |
| Synthetic fine-tuning queries | local doc2query T5 on CPU | Different model from the eval questions; no quota |

Preview models can be withdrawn at short notice, which is another reason caches are committed.

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

**D-EVAL-6 SciFact generation task uses the original labels.** BEIR's SciFact test split is
exactly the original SciFact dev set: all 300 claim ids are present with identical text. The
original release (`scripts/download_scifact_labels.py`, CC BY-NC 2.0, not committed) labels
these claims SUPPORT (124), CONTRADICT (64) and no evidence (112). BEIR still gives the
no-evidence claims a cited abstract as "relevant", but that abstract doesn't settle the claim.
So the generation task is three-way:
- verdict SUPPORTED, REFUTED, or "not found in the provided documents";
- answer correctness is scored against the gold label, without an LLM judge;
- the 112 no-evidence claims are natural "should abstain" cases for the abstention
  precision/recall and false-refusal metrics. This replaces the evidence-ablation idea in
  PLAN.md.

The dev split for τ calibration is the original train set (809 claims, same label scheme).

**D-GEN-1 No reranking in the answer pipeline (decided on the dev split).** Phase 3 was
pre-registered to compare reranked pipelines. On SciFact's dev split (300 train claims), the
MS MARCO MiniLM-L6 cross-encoder lowered quality instead:

| System | nDCG@10 |
|---|---|
| dense bge | 78.1 |
| hybrid BM25 + bge | 75.0 (−3.1\*) |
| hybrid + rerank | 70.8 (−7.3\*\*) |

The test split shows the same pattern (bge 74.0 vs rerank 69.3). So the answer pipelines are
dense bge vs HyDE + bge, and the cross-encoder is used only to score the top-3 abstracts for
the abstention gate.

Caveat: bge scores 78.1 on the train claims vs 74.0 on test, which hints that its training data
may include SciFact's train split. Dev-based choices involving bge carry that caveat.

**D-GEN-2 Retrieval-gate calibration (SciFact).** On the dev split (183 claims with evidence,
117 without), the best threshold with ≤ 5% false refusals is τ = −0.457 (cross-encoder
logit, max over the top-3):
- precision 0.81;
- recall 0.33 of the no-evidence claims.

No-evidence claims usually still retrieve on-topic abstracts, so the gate alone cannot catch
them. The prompt-level abstention rule (layer b) has to do most of that work, and Phase 3
measures the two layers together.

**D-GEN-3 Finding: HyDE's retrieval gain does not reach the answers on SciFact.** 100 claims,
same answerer (gpt-oss-20b), judge (Qwen3) and top-3 context:
- **Retrieval:** dense bge 74.0 vs HyDE + bge (gpt-oss-120b, N=4) 78.5 nDCG@10.
- **Verdict accuracy:** 0.83 vs 0.80, paired Δ −0.03 [−0.08, +0.02], p = 0.45. Only 7 claims
  get a different verdict.
- **Faithfulness:** 0.87 for both (judge, pending human validation).
- **Why:** on the 62 claims with evidence, the gold abstract is already in the top-3 for 92%
  (bge) vs 95% (HyDE). Retrieval is near ceiling at the context depth the answerer uses; the
  remaining errors are the answerer's verdicts and refusals.
- **Cost:** HyDE adds ~2,100 LLM tokens and ~1.5 s per query here, for no answer-level gain.

## Compute

**D-COMP-1 Measured CPU throughput on the dev laptop.** Intel i5-1235U (2 performance + 8
efficiency cores), 16 GB RAM, no GPU, on AC power, measured on SciFact abstracts:

| Model | Default (10 threads, unsorted, batch 32) | Tuned (8 threads, length-sorted, batch 16) |
|---|---|---|
| Contriever | 0.9 docs/s | 2.3 docs/s (full SciFact: 37.6 min) |
| bge-base | 2.0 docs/s | 2.3 docs/s (full SciFact: 37.0 min) |
| MiniLM-L6 cross-encoder | 14 pairs/s | |

Consequences, all at zero cost:
- SciFact encodes locally in about 35 min with Contriever.
- TREC-COVID on the laptop ran at only about 1.3 docs/s under sustained load (thermal
  throttling): about 3 days at 100% CPU. So it was encoded on a free Colab T4 GPU instead
  (D-COMP-3).
- HyDE's query-side cost is visible on CPU: encoding 8 passages takes seconds per query. This
  is reported in the latency columns.

**D-COMP-2 Sanity gate results so far.**

| Dataset | Metric | bm25s | Paper | Pyserini flat |
|---|---|---|---|---|
| SciFact | nDCG@10 | 0.6769 | 67.9 | .6789 |
| SciFact | R@100 | 0.9287 | 92.5 | .9253 |
| TREC-COVID | nDCG@10 | 0.5931 | 59.5 | .5947 |
| TREC-COVID | R@100 | 0.1089 | – | .1091 |
| TREC-COVID | R_cap@100 | 0.471 | 49.8 | – |
| SciFact | Contriever nDCG@10 | 0.6493 | 64.9 | .6493 |
| TREC-COVID | Contriever nDCG@10 | 0.2745 | 27.3 | .2732 |
| TREC-COVID | Contriever R_cap@100 | 0.1722 | 17.2 | – |
| SciFact | BM25 + MiniLM CE nDCG@10 | 0.6881 | – | BEIR paper 0.688 |

Every parity check is within 0.2 points. The paper's TREC-COVID BM25 R_cap 49.8 is BEIR's *Elasticsearch* BM25,
a different system from Pyserini flat, so the 0.471 is not a parity failure.

**D-COMP-3 TREC-COVID corpus encoded on a free Colab GPU (owner decision, 2026-10-15).**
`notebooks/encode_trec_covid_colab.ipynb` mirrors `dense.py` exactly:
- same models and pooling, 512 tokens, `title + " " + text`, corpus order;
- fp32 with TF32 off.

`hyde-rag verify-index` re-encoded 32 random documents per encoder on the laptop CPU and found
cosine ≥ 0.9999998 and max |Δ| < 1e-6 against the T4 vectors, with document order and settings
identical. Queries and hypothetical passages are still encoded on the laptop.

## Licensing

**D-LIC-1** Code: MIT. Dataset and model licences are in `NOTICE`.
`facebook/contriever` is CC BY-NC 4.0 (repo LICENSE; the HF card has no licence field). It is
used only for paper-parity benchmark runs, and the API/demo default to `bge-base-en-v1.5` (MIT).

**D-LIC-2** PDF parsing uses `pdfplumber` (MIT), which gives page numbers plus font size and
name for heading detection. PyMuPDF is avoided because it is AGPL-3.0.

## Food corpus (public data only)

**D-DATA-1 Composition (draft, 43 docs).** `data_sources.yaml` lists the documents, with
licences verified against each publisher's own policy page on 2026-10-14:
- reference codes: FDA Food Code 2026, Codex CXC 1-1969, FSSAI Schedule 4 + 2026 amendment,
  FSSAI recall rules;
- FDA preventive-controls guidance chapters, Listeria, sanitation, seafood/juice HACCP, and
  recall guidance and manuals;
- FSIS compliance guidelines and directives.

35 are committable US public-domain works and 8 are download-only.

**D-DATA-2 "Public domain" is checked per document.** The text of each candidate PDF was
scanned for permission notices. Two flagship US documents reproduce third-party material
"used/reprinted with permission":
- FDA Food Code 2026, Annex 4;
- FSIS Revised Appendix A, p.29.

Both are download-only, as is PCHF Appendix 2 (forms "adapted from" FSPCA), as a precaution.
Codex (CC BY-NC-SA 3.0 IGO) and FSSAI (reuse with acknowledgement, not US public domain) are
download-only under the project rule.

**D-DATA-3 No sensitive documents (owner decision, 2026-10-14).** The corpus excludes
documents about identifiable firms or people:
- FDA warning letters, which name firms and individuals and carry FOIA redactions;
- openFDA and FSIS recall records, which include firm names, addresses and contacts.

Corrective-action and recall questions are answered from guidance instead:
- PCHF Ch 5 and Ch 14;
- FDA's low-moisture RTE sanitation and corrective-action guidance;
- FDA recall guidance, RPM Ch 7 and IOM Ch 7;
- FSIS Directive 8080.1;
- FSSAI recall regulations.

**D-DATA-4 Polite, reproducible downloading.**
- fda.gov: `Crawl-Delay: 30` plus bot detection, so downloads are single-threaded with ≥30 s
  between requests and back-off on errors.
- usda.gov/fsis.usda.gov: these return 403 to scripted clients from the dev network, so the
  script falls back to "download manually to this path" and verifies sha256. These files are
  public domain, so once committed nobody needs to re-download them.
- We never try to evade bot detection.

**D-DATA-5 FSSAI Schedule 4 is amended.** The 2021 compendium's Schedule 4 is partly
superseded by Amendment No. 5 (23 June 2026), so both are included and eval questions cite the
amended text where they differ.

## Domain adaptation

**D-FT-1 F2 = bge-base fine-tuned on synthetic queries, on a free Colab GPU.**
- Training data comes from corpus chunks only. Each chunk gets 3 sampled queries from
  `doc2query/msmarco-t5-base-v1` (Apache-2.0). That is a different model and prompt from the
  eval-set drafts (gpt-oss-20b), and no eval question or gold span is ever used, so F2 cannot
  memorise the test set.
- Hard negatives: one random chunk ranked 10-50 by base bge-base for each query, skipping the
  positive's own page (likely false negatives). Loss: MultipleNegativesRankingLoss with in-batch
  negatives; 1 epoch, lr 2e-5, batch 64, no-duplicates sampler; the BGE query instruction goes on
  anchors only, matching inference.
- The training code is one tested module (`src/hyde_rag/train/finetune_core.py`), embedded
  verbatim in `notebooks/finetune_food_embedder_colab.ipynb`, so the notebook cannot drift from
  the repo. The notebook also encodes the corpus with base and tuned models; `verify-index`
  re-encodes a sample locally and checks it against the Colab vectors.
- The tuned model and the synthetic queries are gitignored (derived from documents that are not all
  public domain); the notebook regenerates them.

**D-FT-2 F1 design: generic vs domain instruction at N=1, domain at N=4 as the ablation.**
On BEIR, HyDE-bge at N=1 matched N=4 (SciFact 78.6 vs 78.5), so the instruction comparison
(`paper.web_search` vs `domain.food_quality_v1`) runs at N=1, and only the domain instruction
gets N=4 (query weight 0.5, D-HYDE-8). This halves the gpt-oss-120b quota versus N=4 for both
(about 4.7 free-tier days instead of 7 for 329 questions). Systems: `configs/experiments/food-test.yaml`;
`food-dev.yaml` is used only to calibrate τ and pick the answer pipeline's mode.

**D-FT-3 HyDE passages are generated during the human review.** `generate --pending-review`
uses every non-rejected question in `review.csv`, with edits applied. Cached outputs are found by
request key (the question text is part of the request), so an edited question is simply regenerated
and never picks up passages for its old wording (regression test in `tests/test_experiment.py`).
The cost of generating for questions that are later rejected is free-tier quota, not money.

**D-EVAL-7 The food eval set is LLM-reviewed and human-audited, not human-reviewed (owner decision,
2026-10-21).** The owner asked Claude (Opus 5.5, the coding assistant on this project) to review the
329 drafted questions instead of doing it by hand.
- Every drafted row got accept / edit / reject with a written reason. These are logged without
  document text in `eval_sets/food_quality/review_decisions.jsonl`, and the notes are tagged
  `[llm-review]` in `review.csv`.
- Rules were applied as in `REVIEW_GUIDE.md`:
  - reject bibliography, boilerplate and metadata questions, contrived multi-document combinations,
    near-duplicates, and answers that misread tables;
  - reject an "unanswerable" row whenever any document partly answers it (checked with corpus
    search);
  - name the source ("FDA's PCHF guidance", "FSSAI Schedule 4") wherever "the guidance" was
    ambiguous in a 41-document corpus.
- Every replaced evidence quote was machine-checked with the same matcher `build-evalset` uses.
  Cross-batch near-duplicates were found with bge-base cosine > 0.88 and resolved by hand.
- Result: 131 accepted, 74 edited, 124 rejected, giving 205 items (62 dev / 143 test, 15
  unanswerable). No answerable item was excluded for missing evidence.
- A person audits a random sample of 30 kept and 10 rejected decisions (`review-audit-sheet`,
  `review-audit-agreement`). The agreement rate is reported with the food results, and the eval
  set is described as "LLM-reviewed, human-audited". If the audit disagrees often, the review is
  redone by hand.
- Commit policy: `dev.jsonl` / `test.jsonl` (questions, short reference answers, evidence as
  character spans) and `review_decisions.jsonl` are committed. `review.csv`, `candidates.jsonl` and
  `audit.csv` hold longer verbatim excerpts, some from non-US-federal documents (Codex, FSSAI), so
  they stay local (D-DATA-1). 262 of 289 gold evidence spans are in FDA/FSIS documents.

**D-DATA-6 Known corpus defect: garbled glyphs.** 20 of 3,890 chunks (16 in `fssai_recall_reg_2017`,
4 in `fda_retail_haccp_manual`) contain unmapped font glyphs (`(cid:NNNN)`) from Hindi/bilingual
PDF pages. They stay in the index; questions drafted from them were rejected in review. Fixing it
needs OCR or a different extractor for those pages.
