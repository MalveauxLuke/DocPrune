# ColQwen Top-5 Run Inspection Report

## Verdict
PASS WITH WARNINGS

The run completed cleanly and the `top5_results.jsonl` file is structurally
usable for binary-classifier table construction. Retrieval quality is weak as a
standalone positive source: only 454 / 4000 queries have the gold page in the
top 5. Missing gold pages should be force-added before Baseline 1 or
cross-attention training so every query has a positive example.

## Run
- Job: `57003877`
- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `03:14:51`
- Batch MaxRSS: `17175928K`
- Run dir: `/scratch/lmalveau/mmdocir_colqwen_diverse_longdoc_runs/20260624T055236Z`
- Result file: `/scratch/lmalveau/mmdocir_colqwen_diverse_longdoc_runs/20260624T055236Z/top5_results.jsonl`
- Model: `vidore/colqwen2.5-v0.1`

No traceback, OOM, import failure, killed job, or fatal error signature was
found in `logs/colqwen_top5_57003877.err` or `.out`. The stderr includes
Hugging Face unauthenticated-download warnings and ColQwen load-report
missing/unexpected LoRA key warnings, but the job finished and wrote all rows.

## Dataset Counts
- Queries: `4000`
- Corpus pages: `6817`
- PNG pages referenced by corpus: `6817`
- PNG pages found on disk: `6817`
- Missing PNG pages: `0`
- Missing gold pages from corpus: `0`

Query counts by source dataset:

| Domain | Queries |
| --- | ---: |
| ArxivQA | 400 |
| DUDE_long | 300 |
| MP-DocVQA | 1000 |
| SciQAG | 300 |
| SlideVQA | 600 |
| TAT-DQA | 1400 |

## Result Integrity
- JSONL rows: `4000`
- Candidate-count distribution: `5 candidates` for `4000` rows
- Missing query ids: `0`
- Extra query ids: `0`
- Duplicate query ids: `0`
- Rows with scores out of descending order: `0`
- Rows with invalid rank sequence: `0`
- Rows with duplicate candidates: `0`

## Gold Coverage
- Gold in top 5: `454 / 4000` (`11.35%`)
- Gold not in top 5: `3546 / 4000` (`88.65%`)

Gold rank distribution when present:

| Rank | Count |
| ---: | ---: |
| 1 | 201 |
| 2 | 91 |
| 3 | 70 |
| 4 | 57 |
| 5 | 35 |

Gold is not rank 1 in `253 / 454` gold-in-top-5 cases (`55.73%`). There are
`542` total non-gold candidates ranked above gold across those cases.

Per-domain gold-in-top-5 rates:

| Domain | Hits | Total | Rate |
| --- | ---: | ---: | ---: |
| ArxivQA | 75 | 400 | 18.75% |
| DUDE | 22 | 300 | 7.33% |
| MP-DocVQA | 83 | 1000 | 8.30% |
| SciQAG | 153 | 300 | 51.00% |
| SlideVQA | 84 | 600 | 14.00% |
| TAT-DQA | 37 | 1400 | 2.64% |

## Training Readiness
The outputs are suitable for hard-negative mining and classifier-table input
after adding the gold page for every query. Do not train directly from top-5
rows as-is unless the table builder force-adds missing positives; otherwise
`3546` queries would have no positive candidate. The low TAT-DQA, DUDE, and
MP-DocVQA hit rates are the main quality warning, not a file-integrity blocker.

No blocker remains before Baseline 1 or cross-attention training if the table
builder enforces one positive per query and keeps the retrieved top-5 pages as
hard negatives.
