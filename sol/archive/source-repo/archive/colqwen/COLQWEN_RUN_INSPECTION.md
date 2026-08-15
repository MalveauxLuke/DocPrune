# ColQwen Document-Scoped Run Inspection

## Task
Inspect the corrected document-scoped ColQwen top-5 run and write a short
readiness report for classifier training.

## Context
- Repo on SOL: `~/COLPALI_binary_classification`
- Dataset: `/scratch/$USER/mmdocir_colqwen_diverse_longdoc_pilot`
- Output root: `/scratch/$USER/mmdocir_colqwen_docscoped_runs`
- Result file: `doc_top5_results.jsonl`
- Expected counts: `4000` queries, `6817` corpus pages, `6817` PNG pages

## Check
1. Job completed successfully with no traceback/OOM/import failure.
2. Result file exists and has `4000` JSONL rows.
3. Every query appears once, with no missing/extra/duplicate query ids.
4. Each row has 5 ranked candidates, sorted by descending score.
5. Every candidate belongs to the query's annotated gold document.
6. Dataset counts still match the expected counts.
7. Retrieval quality is suitable for training analysis.

## Report
Write findings to:

```text
sol/COLQWEN_DOCSCOPED_RUN_INSPECTION_REPORT.md
```

Include:
- verdict: `PASS`, `PASS WITH WARNINGS`, or `FAIL`
- inspected run dir and result file
- job state and exit code
- dataset counts
- result row count and candidate-count distribution
- query coverage issues, if any
- any candidates outside the query's gold document
- gold-in-top-1/top-3/top-5 count/rate
- gold rank distribution
- percent of gold-in-top-5 cases where gold is not rank 1
- per-domain gold-in-top-5 rates
- number of cases where a non-gold candidate outranks gold when gold is present
- whether any missing gold pages must be force-added for classifier training
- any blockers before Baseline 1 or cross-attention training

Do not run `git pull`, `git push`, or create a pull request. When the report is
ready, ask the user in chat to handle any needed repo sync or publishing.
