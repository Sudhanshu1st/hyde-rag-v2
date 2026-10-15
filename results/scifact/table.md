### scifact (scifact test, 300 queries)

nDCG@10 with 95% bootstrap CI; Δ = paired difference vs the system's naive-dense baseline (Holm-adjusted randomisation-test p). CPU ms = single-query retrieval on the dev laptop (no LLM); LLM s and tokens are per query (N samples in parallel).

| System | nDCG@10 [95% CI] | Δ nDCG@10 (p) | R@10 | R@100 | MRR@10 | CPU ms | LLM s | LLM tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|
| bm25 | 67.7 [63.2, 72.0] | +2.8 (p=0.491) | 80.3 | 92.9 | 64.2 | 1 | - | - |
| dense-contriever | 64.9 [60.3, 69.5] | baseline | 78.9 | 92.6 | 61.1 | 72 | - | - |
| dense-bge | 74.0 [70.0, 77.9] | baseline | 87.4 | 96.7 | 70.3 | 73 | - | - |
| hyde-contriever-20b-n1 | 65.3 [60.8, 69.7] | +0.3 (p=1.000) | 78.8 | 94.1 | 61.8 | 988 | 1.0 | 615 |
| hyde-bge-20b-n1 | 77.8 [74.1, 81.4] | +3.7** (p=0.001) | 91.3 | 97.0 | 74.0 | 705 | 1.0 | 615 |
| hybrid-contriever | 69.7 [65.7, 73.7] | +4.8** (p=0.004) | 86.1 | 96.1 | 65.2 | 73 | - | - |
| hybrid-bge | 74.1 [70.0, 77.9] | +0.0 (p=1.000) | 87.3 | 97.7 | 70.5 | 74 | - | - |
| rerank-bm25 | 68.8 [64.3, 73.1] | +3.9 (p=0.139) | 80.4 | 92.9 | 66.0 | 8,196 | - | - |
| rerank-dense-contriever | 69.8 [65.5, 74.0] | +4.9* (p=0.024) | 83.4 | 92.6 | 66.4 | 7,168 | - | - |
| rerank-dense-bge | 69.3 [65.0, 73.6] | -4.7* (p=0.011) | 82.2 | 96.7 | 66.1 | 7,912 | - | - |
| rerank-hybrid-bge | 68.7 [64.2, 73.0] | -5.4** (p=0.005) | 80.6 | 97.7 | 65.7 | 8,065 | - | - |

\* p < 0.05, \*\* p < 0.01 after Holm correction.
