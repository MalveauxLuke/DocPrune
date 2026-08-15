# ColQwen Document-Scoped Top-5 Inspection Report

## Verdict
PASS WITH WARNINGS

The corrected document-scoped run completed successfully and every retrieved
candidate stayed within the query's annotated gold document. The output is
usable for classifier-table construction, but gold still appears in the top 5
for only `2088 / 4000` queries (`52.20%`), so the table builder should
force-add the gold page for the remaining `1912` queries.

## Run
- Job: `57088606`
- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `03:44:48`
- Batch MaxRSS: `16847884K`
- Run dir: `/scratch/lmalveau/mmdocir_colqwen_docscoped_runs/20260624T200158Z`
- Result file: `/scratch/lmalveau/mmdocir_colqwen_docscoped_runs/20260624T200158Z/doc_top5_results.jsonl`
- Scope: `document`

No traceback, OOM, killed job, fatal error, or failed-job signature was found in
the job logs. The output log reports all `4000 / 4000` queries scored.

## Dataset And Result Integrity
- Queries: `4000`
- Corpus pages: `6817`
- PNG pages found from corpus references: `6817`
- JSONL rows: `4000`
- Candidate-count distribution: `5 candidates` for `4000` rows
- Retrieval-scope distribution: `document` for `4000` rows
- Missing query ids: `0`
- Extra query ids: `0`
- Duplicate query ids: `0`
- Rows with invalid rank sequence: `0`
- Rows with scores out of descending order: `0`
- Rows with duplicate candidates: `0`
- Rows with candidates outside the annotated gold document: `0`
- Missing gold pages from corpus: `0`

## Gold Coverage
- Gold in top 1: `744 / 4000` (`18.60%`)
- Gold in top 3: `1554 / 4000` (`38.85%`)
- Gold in top 5: `2088 / 4000` (`52.20%`)
- Gold absent from top 5: `1912 / 4000` (`47.80%`)

Gold rank distribution:

| Rank | Count |
| ---: | ---: |
| 1 | 744 |
| 2 | 453 |
| 3 | 357 |
| 4 | 294 |
| 5 | 240 |

Across gold-present rows, `3009` non-gold candidates outrank gold in aggregate.

## Per-Domain Coverage

| Domain | Total | Top 1 | Top 3 | Top 5 |
| --- | ---: | ---: | ---: | ---: |
| ArxivQA | 400 | 104 (26.00%) | 212 (53.00%) | 288 (72.00%) |
| DUDE | 300 | 69 (23.00%) | 151 (50.33%) | 207 (69.00%) |
| MP-DocVQA | 1000 | 132 (13.20%) | 309 (30.90%) | 416 (41.60%) |
| SciQAG | 300 | 124 (41.33%) | 215 (71.67%) | 276 (92.00%) |
| SlideVQA | 600 | 173 (28.83%) | 341 (56.83%) | 450 (75.00%) |
| TAT-DQA | 1400 | 142 (10.14%) | 326 (23.29%) | 451 (32.21%) |

## Training Readiness
Proceed to classifier-table construction with forced positive inclusion. Use
the document-scoped top-5 candidates as hard negatives, but do not assume the
retrieved top 5 contains a positive for every query. The main remaining quality
warning is low top-5 gold coverage for TAT-DQA and MP-DocVQA.
