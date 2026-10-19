### scifact-dev (scifact train, 300 queries)

nDCG@10 with 95% bootstrap CI; Δ = paired difference vs the system's naive-dense baseline (Holm-adjusted randomisation-test p). CPU ms = single-query retrieval on the dev laptop (no LLM); LLM s and tokens are per query (N samples in parallel).

| System | nDCG@10 [95% CI] | Δ nDCG@10 (p) | R@10 | R@100 | MRR@10 | CPU ms | LLM s | LLM tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|
| bm25 | 68.0 [63.6, 72.2] | -10.1** (p=0.000) | 81.0 | 93.0 | 64.7 | 0 | - | - |
| dense-bge | 78.1 [74.3, 81.8] | baseline | 90.2 | 97.3 | 75.1 | 82 | - | - |
| hybrid-bge | 75.0 [71.1, 78.8] | -3.1* (p=0.019) | 87.0 | 97.2 | 72.3 | 82 | - | - |
| rerank-hybrid-bge | 70.8 [66.5, 74.9] | -7.3** (p=0.000) | 83.7 | 97.2 | 67.6 | 11,198 | - | - |

\* p < 0.05, \*\* p < 0.01 after Holm correction.
