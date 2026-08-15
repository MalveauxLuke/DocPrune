# ColQwen2 Document-Scoped Top-5 Inspection Report

## Verdict
PASS

The validated ColQwen2 path (`model_arch=colqwen2`,
`model_name=vidore/colqwen2-v1.0`) completed the full MMDocIR document-scoped
top-5 run successfully. Retrieval quality is strong and far better than the
bad ColQwen2.5 document-scoped run.

## Run
- Job id: `57151538`
- Job name: `colqwen2-doc-top5`
- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:29:25`
- Batch MaxRSS: `45489968K`
- Output directory: `/scratch/lmalveau/mmdocir_colqwen2_docscoped_runs/20260625T035758Z`
- Result file: `/scratch/lmalveau/mmdocir_colqwen2_docscoped_runs/20260625T035758Z/doc_top5_results.jsonl`
- Rows: `4000`

## Gold Coverage

| Metric | Count | Rate |
| --- | ---: | ---: |
| Gold@1 | 2811 / 4000 | 70.28% |
| Gold@3 | 3675 / 4000 | 91.88% |
| Gold@5 | 3889 / 4000 | 97.22% |

Gold rank distribution:

| Rank | Count |
| ---: | ---: |
| 1 | 2811 |
| 2 | 640 |
| 3 | 224 |
| 4 | 147 |
| 5 | 67 |
| Absent from top 5 | 111 |

Non-gold candidates outranking gold when gold is present: `1797`.

## Per-Domain Gold Coverage

| Domain | Total | Gold@1 | Gold@3 | Gold@5 |
| --- | ---: | ---: | ---: | ---: |
| ArxivQA | 400 | 312 (78.00%) | 379 (94.75%) | 395 (98.75%) |
| DUDE | 300 | 216 (72.00%) | 270 (90.00%) | 289 (96.33%) |
| MP-DocVQA | 1000 | 647 (64.70%) | 867 (86.70%) | 935 (93.50%) |
| SciQAG | 300 | 178 (59.33%) | 260 (86.67%) | 293 (97.67%) |
| SlideVQA | 600 | 439 (73.17%) | 583 (97.17%) | 597 (99.50%) |
| TAT-DQA | 1400 | 1019 (72.79%) | 1316 (94.00%) | 1380 (98.57%) |

## Integrity Checks

| Check | Count |
| --- | ---: |
| Duplicate query ids | 0 |
| Missing query ids | 0 |
| Extra query ids | 0 |
| Invalid rank rows | 0 |
| Unsorted score rows | 0 |
| Duplicate candidate rows | 0 |
| Candidates outside gold document | 0 |
| Missing gold pages | 0 |
| Mismatched gold flags | 0 |

## Comparison
The ColQwen2 document-scoped run is dramatically better than the bad ColQwen2.5
document-scoped run:

| Run | Gold@1 | Gold@3 | Gold@5 |
| --- | ---: | ---: | ---: |
| ColQwen2 doc-scoped | 70.28% | 91.88% | 97.22% |
| ColQwen2.5 doc-scoped | 18.60% | 38.85% | 52.20% |

## Conclusion
The validated ColQwen2 path is healthy on MMDocIR document-scoped retrieval.
The previous ColQwen2.5 result remains suspect due to model/path compatibility,
not because document-scoped retrieval or the dataset materialization is broken.
