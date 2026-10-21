### food-dev (food dev, 62 queries)

nDCG@10 with 95% bootstrap CI; Δ = paired difference vs the system's naive-dense baseline (Holm-adjusted randomisation-test p). CPU ms = single-query retrieval on the dev laptop (no LLM); LLM s and tokens are per query (N samples in parallel).

| System | nDCG@10 [95% CI] | Δ nDCG@10 (p) | R@10 | R@100 | MRR@10 | CPU ms | LLM s | LLM tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|
| bm25 | 75.1 [65.8, 83.7] | +16.2* (p=0.037) | 88.8 | 100.0 | 71.6 | 0 | - | - |
| dense-bge | 58.8 [49.0, 68.8] | baseline | 75.9 | 96.6 | 54.6 | 79 | - | - |
| hybrid-bge | 68.6 [59.9, 77.1] | +9.8* (p=0.013) | 90.5 | 98.3 | 62.7 | 79 | - | - |
| hyde-bge-web-n1 | 59.5 [49.3, 70.0] | +0.7 (p=1.000) | 72.4 | 96.6 | 56.6 | 311 | 1.0 | 396 |
| hyde-bge-domain-n1 | 59.4 [49.0, 69.5] | +0.6 (p=1.000) | 74.1 | 96.6 | 55.9 | 395 | 1.1 | 458 |
| hybrid-hyde-bge-domain | 69.3 [60.4, 78.2] | +10.5* (p=0.014) | 88.8 | 98.3 | 64.2 | 395 | 1.1 | 458 |
| dense-bge-food | 65.2 [55.1, 74.8] | +6.4 (p=0.418) | 80.2 | 93.1 | 61.0 | 83 | - | - |
| hyde-bge-food-domain-n1 | 58.5 [48.6, 68.2] | -6.8* (p=0.031) | 75.9 | 96.6 | 53.7 | 389 | 1.1 | 458 |
| rerank-dense-bge | 78.7 [70.0, 86.5] | +19.9** (p=0.004) | 89.7 | 96.6 | 75.9 | 12,755 | - | - |
| rerank-hyde-bge-domain-n1 | 78.7 [70.0, 86.5] | +19.9** (p=0.004) | 89.7 | 96.6 | 75.9 | 14,363 | 1.1 | 458 |

\* p < 0.05, \*\* p < 0.01 after Holm correction.
