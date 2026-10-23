### food-test (food test, 143 queries)

nDCG@10 with 95% bootstrap CI; Δ = paired difference vs the system's naive-dense baseline (Holm-adjusted randomisation-test p). CPU ms = single-query retrieval on the dev laptop (no LLM); LLM s and tokens are per query (N samples in parallel).

| System | nDCG@10 [95% CI] | Δ nDCG@10 (p) | R@10 | R@100 | MRR@10 | CPU ms | LLM s | LLM tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|
| bm25 | 74.6 [69.2, 79.7] | +8.2 (p=0.097) | 93.6 | 99.6 | 69.4 | 0 | - | - |
| dense-bge | 66.5 [60.2, 72.4] | baseline | 86.4 | 98.9 | 60.9 | 166 | - | - |
| hybrid-bge | 78.1 [73.0, 82.9] | +11.6** (p=0.002) | 94.3 | 100.0 | 73.6 | 166 | - | - |
| hyde-bge-web-n1 | 65.1 [58.4, 71.3] | -1.4 (p=1.000) | 83.0 | 99.2 | 60.1 | 813 | 0.9 | 376 |
| hyde-bge-domain-n1 | 67.1 [60.6, 73.3] | +0.7 (p=1.000) | 84.5 | 97.3 | 62.2 | 1,218 | 1.1 | 455 |
| hyde-bge-domain-n4-qw50 | 67.3 [61.0, 73.4] | +0.8 (p=1.000) | 85.2 | 99.2 | 62.2 | 4,397 | 1.4 | 1,862 |
| hybrid-hyde-bge-domain | 77.3 [71.9, 82.5] | +10.9** (p=0.002) | 93.2 | 99.6 | 72.8 | 1,218 | 1.1 | 455 |
| bm25-q2d-domain | 69.9 [63.8, 75.7] | +3.5 (p=1.000) | 88.3 | 99.6 | 65.0 | 2 | 1.1 | 455 |
| mq-hybrid-bge | 72.8 [67.0, 78.1] | +6.3 (p=0.051) | 91.3 | 100.0 | 67.8 | 720 | 0.6 | 211 |
| dense-bge-food | 66.0 [59.8, 71.8] | -0.5 (p=1.000) | 86.4 | 99.2 | 60.1 | 177 | - | - |
| hyde-bge-food-domain-n1 | 66.8 [60.6, 72.7] | +0.9 (p=1.000) | 85.6 | 98.1 | 61.5 | 1,275 | 1.1 | 455 |
| hybrid-bge-food | 77.7 [72.6, 82.4] | +11.2** (p=0.002) | 95.8 | 99.6 | 72.8 | 177 | - | - |
| rerank-dense-bge | 80.1 [74.9, 85.0] | +13.7** (p=0.002) | 94.3 | 98.9 | 76.3 | 26,321 | - | - |
| rerank-hybrid-bge | 80.5 [75.4, 85.1] | +14.0** (p=0.002) | 95.1 | 100.0 | 76.5 | 24,815 | - | - |
| rerank-hyde-bge-domain-n1 | 78.8 [73.5, 83.8] | +12.4** (p=0.002) | 93.6 | 97.3 | 74.8 | 26,815 | 1.1 | 455 |
| rerank-dense-bge-food | 80.7 [75.7, 85.2] | +14.2** (p=0.002) | 96.2 | 99.2 | 76.3 | 25,305 | - | - |
| rerank-hyde-bge-food-domain-n1 | 80.6 [75.8, 85.1] | +14.2** (p=0.002) | 95.8 | 98.1 | 76.4 | 25,914 | 1.1 | 455 |
| rerank-hybrid-bge-d20 | 81.7 [76.9, 86.0] | +15.2** (p=0.002) | 97.0 | 99.2 | 77.4 | 6,103 | - | - |

\* p < 0.05, \*\* p < 0.01 after Holm correction.
