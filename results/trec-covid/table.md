### trec-covid (trec-covid test, 50 queries)

nDCG@10 with 95% bootstrap CI; Δ = paired difference vs the system's naive-dense baseline (Holm-adjusted randomisation-test p). CPU ms = single-query retrieval on the dev laptop (no LLM); LLM s and tokens are per query (N samples in parallel).

| System | nDCG@10 [95% CI] | Δ nDCG@10 (p) | R@10 | R@100 | MRR@10 | CPU ms | LLM s | LLM tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|
| bm25 | 59.3 [51.9, 66.6] | +31.9** (p=0.001) | 1.6 | 10.9 | 85.6 | 2 | - | - |
| dense-contriever | 27.4 [20.8, 34.7] | baseline | 0.7 | 3.7 | 54.1 | 2,045 | - | - |
| dense-bge | 78.1 [72.4, 83.3] | baseline | 2.2 | 14.1 | 91.8 | 310 | - | - |
| hyde-contriever-20b-n1 | 55.1 [46.3, 63.7] | +27.6** (p=0.001) | 1.4 | 8.5 | 78.7 | 760 | 1.0 | 672 |
| hyde-contriever-qwen-n4 | 54.1 [45.2, 62.9] | +26.6** (p=0.001) | 1.5 | 8.6 | 73.9 | 1,833 | 0.9 | 1,378 |
| hyde-contriever-qwen-n4-qw50 | 60.9 [52.3, 69.4] | +33.5** (p=0.001) | 1.7 | 9.8 | 80.4 | 2,063 | 0.9 | 1,378 |
| hybrid-contriever | 57.8 [50.1, 65.5] | +30.4** (p=0.001) | 1.5 | 9.0 | 81.8 | 2,047 | - | - |
| hybrid-bge | 80.6 [75.0, 85.9] | +2.5 (p=0.867) | 2.2 | 14.6 | 95.3 | 312 | - | - |
| rerank-bm25 | 73.7 [67.2, 80.0] | +46.3** (p=0.001) | 2.0 | 10.9 | 87.0 | 10,148 | - | - |
| rerank-dense-bge | 77.1 [70.9, 83.1] | -1.0 (p=0.867) | 2.1 | 14.1 | 87.7 | 9,636 | - | - |
| rerank-hybrid-bge | 75.6 [69.4, 81.6] | -2.5 (p=0.867) | 2.1 | 14.6 | 87.3 | 9,963 | - | - |

\* p < 0.05, \*\* p < 0.01 after Holm correction.
