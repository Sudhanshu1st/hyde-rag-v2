# Does HyDE still help in 2026? I measured it.

*Draft. Every number is from [`docs/RESULTS.md`](../RESULTS.md), which is generated from the
committed runs. Food-safety numbers marked ⟨pending⟩ are still being computed.*

HyDE (Hypothetical Document Embeddings, Gao et al., ACL 2023) is one of those RAG tricks that
shows up in every tutorial:

1. ask an LLM to *write* a passage that would answer the question;
2. embed that fake passage;
3. search with it.

The fake passage may be wrong, but it *looks like* the documents you want, so it lands closer to
them in embedding space than a short question does.

The paper paired it with Contriever, an unsupervised 2022 embedder, and `text-davinci-003`. Three
years on, embedders are trained on hundreds of millions of question-passage pairs, and the best
open LLMs are free to call. So I wanted to know:

- Does HyDE still help a modern embedder?
- How many hypothetical documents do you need?
- Does any of it hold outside academic benchmarks, on the kind of documents a food company's
  quality team actually searches?

I rebuilt everything from scratch and ran it for $0.

## Setup

- **Retrievers:**
  - BM25 (bm25s, configured like Pyserini);
  - Contriever, the paper's encoder;
  - `bge-base-en-v1.5` as the modern supervised embedder;
  - hybrid (RRF), Query2doc, multi-query (RAG-Fusion), and a MiniLM cross-encoder reranker.
- **Generators:** `gpt-oss-120b`, `gpt-oss-20b` and `qwen3.8-27b`, all on Groq's free tier.
  Every output is cached and committed, so anyone can re-score without an API key.
- **Benchmarks:**
  - SciFact (300 claims) and TREC-COVID (50 queries) from BEIR;
  - a food-safety set I built: 41 public documents (FDA Food Code and guidance, USDA FSIS, Codex,
    FSSAI), 3,890 chunks and 205 reviewed questions (62 dev, 143 test).
- **Statistics:** paired bootstrap confidence intervals, a paired randomisation test against the
  same encoder without HyDE, and Holm correction. Every design choice is logged in
  [`DECISIONS.md`](../DECISIONS.md).

## 1. The paper reproduces

| | SciFact (ours / paper) | TREC-COVID (ours / paper) |
|---|---:|---:|
| BM25 | 67.7 / 67.9 | 59.3 / 59.5 |
| Contriever | 64.9 / 64.9 | 27.4 / 27.3 |
| HyDE + Contriever, N=1 | 67.4 / 69.1 | 56.8 / 59.3 |

nDCG@10. The baselines land within 0.2 points of the paper, and Contriever's TREC-COVID capped
recall@100 is exactly the reported 17.2. HyDE with a free open model gets within 1.7–2.5 points of
`text-davinci-003`.

## 2. The paper's averaging quietly breaks when you sample more

The paper averages the question with N hypothetical documents (its Eq. 8):
v = (f(q) + Σ f(d̂ₖ)) / (N+1).

At N=4 the question is only 20% of the vector. With Contriever that hurts:

- SciFact drops from 67.4 at N=1 to **61.0** at N=4, below plain Contriever.
- TREC-COVID drops from 56.8 to **46.9**.

Diagnosis:
- The individual samples are fine; any single one scores 0.66–0.67 on SciFact.
- The passages are weaker queries than the claim itself, so the more of them you average in, the
  more you dilute the one signal that is precisely on topic.

Keeping the question at a fixed 50% fixes it:
- **+6.7** nDCG@10 on SciFact [95% CI +4.6, +8.9];
- **+12.0** on TREC-COVID [+7.9, +16.3].

I picked the 50% weight on SciFact, so TREC-COVID is an out-of-sample check.

## 3. HyDE still helps a modern embedder, if you don't dilute the question

| bge-base | SciFact | TREC-COVID |
|---|---:|---:|
| bge alone | 74.0 | 78.1 |
| HyDE, N=1 | **78.6** (+4.5, p=0.003) | 83.4 (+5.3, p=0.07) |
| HyDE, N=4, Eq. 8 averaging | 78.5 | **77.2** (gain gone) |
| HyDE, N=4, question at 50% | 77.8 | **84.8** (+6.7, p=0.002) |

Supervised embedding didn't make HyDE obsolete. On TREC-COVID, though, a naive N=4 setup
reproduces the original averaging and makes HyDE look useless. If you've read "HyDE doesn't help
with modern embedders," check how the vectors were averaged.

## 4. One hypothetical document is enough

N=4 with the 50% question weight is within noise of N=1 for both encoders on both datasets (−0.8
to +2.1, all p > 0.1), and costs four times the tokens: 2,099 vs 521 per query. Use N=1.

## 5. Cheaper tricks get close

- **Multi-query** (four LLM rewrites fused with BM25 + bge, about 200 tokens per query) matches
  HyDE on TREC-COVID (83.7 vs 84.8, n.s.) but trails on SciFact (75.8 vs 78.6, p = 0.03).
- **Hybrid BM25 + bge** with no LLM at all reaches 80.6 on TREC-COVID.
- **A small cross-encoder hurts strong retrievers:** reranking bge's top 100 with MiniLM-L6 drops
  SciFact from 74.0 to 69.3.

## 6. Better retrieval didn't mean better answers on SciFact

I also generated cited answers (gpt-oss-20b, top-3 context, a two-layer "not found" gate) for 100
SciFact claims, and had a different model judge them.

- HyDE's retrieval gain did not show up in the answers: verdict accuracy 0.80 vs 0.83 without
  HyDE (n.s.), faithfulness 0.87 for both.
- The reason is that the gold evidence was already in the top 3 for 92–95% of claims, so the
  answers weren't limited by retrieval.

## 7. Food-safety documents: a different picture

This is the part I cared about most. Quality managers ask questions like "Under FSIS Listeria
guidance, if food contact surfaces test positive on day 7, which lots must be held?" and want an
answer with a citation they can check.

On the food dev split (62 questions):

| System | nDCG@10 |
|---|---:|
| BM25 | 75.1 |
| bge | 58.8 |
| HyDE + bge (domain or generic instruction) | 59.4–59.5 (+0.6, n.s.) |
| bge fine-tuned on synthetic queries (F2) | 65.2 |
| HyDE + fine-tuned bge | 58.5 (−6.8 vs fine-tuned alone) |
| bge + cross-encoder rerank | **78.7** (+19.9, p = 0.004) |

Test split (143 questions): ⟨pending⟩.

Every lesson from the scientific benchmarks flips:
- HyDE does nothing.
- Keyword search beats the dense embedder.
- Reranking, which hurt on SciFact, is the biggest win.

Why this happens is a hypothesis, not a result. Regulatory questions turn on exact terms (part
numbers, named programs, temperatures and times). A hypothetical passage invents plausible
specifics, and an embedding of plausible specifics isn't much closer to the *right* specifics than
the question was. BM25 and the cross-encoder both reward exact overlap.

Answers on the food set (plain bge vs bge + rerank, 100 test questions, graded against reviewed
reference answers): ⟨pending⟩.

## What I'd tell a team building RAG today

1. **Measure on your own documents first.** HyDE's sign flipped between scientific abstracts and
   regulatory guidance. A 60-question dev set was enough to see that.
2. **If you use HyDE, use one hypothetical document,** and never let the question drop below about
   half of the vector.
3. **Start with BM25 + dense hybrid, then try a cross-encoder.** Check the reranker on your data;
   it was the best and worst idea here, depending on the corpus.
4. **Retrieval gains don't automatically reach the answers.** Evaluate the answers, not just
   nDCG, and check your LLM judge against a few dozen human labels before trusting it.
5. **None of this needs a budget.** Every LLM call here ran on free tiers, and the GPU work (a
   171K-document encoding and an embedder fine-tune) ran on free Colab T4s.

## Caveats

- TREC-COVID has 50 queries and the food dev split 62, so the confidence intervals are wide.
- The food question set was drafted by an LLM and reviewed by an LLM (Claude). A human audit of
  40 random review decisions agreed with all 40.
- The answer judge is an LLM, and it is too strict. On 49 human-labelled claims it agreed with the
  person only 55% of the time (κ = 0.12), because people overturned almost all of its "not
  supported" verdicts. Its faithfulness scores are lower bounds; re-weighted by the human labels,
  SciFact faithfulness is about 0.98, not 0.87. Validate your judge before trusting its numbers:
  that alone changed my conclusion about how faithful the answers were.
- I tested one modern embedder family (bge). Results for others may differ.

Code, data scripts, cached generations and every number:
<https://github.com/Sudhanshu1st/hyde-rag-v2>.
