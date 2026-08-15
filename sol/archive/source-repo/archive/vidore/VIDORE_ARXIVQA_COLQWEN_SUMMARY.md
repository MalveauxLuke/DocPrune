# ViDoRe ArxivQA ColQwen Summary

## Verdict
RUN COMPLETED, METRIC MISMATCH

The ViDoRe ArxivQA ColQwen2.5 calibration job completed end to end after fixing
the missing `datasets` dependency. The local/SOL run produced valid output
files, but its `nDCG@5` is far below the published ColPali-style benchmark
reference, so the pipeline should be investigated before using this benchmark
as a calibration pass.

## Jobs
- Failed job: `57128552`
- Failure: `ModuleNotFoundError: No module named 'datasets'`
- Fix note: `sol/bug_fixes/VIDORE_ARXIVQA_DATASETS_FIX.md`
- Successful job: `57129343`
- Successful job state: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:16:44`
- Batch MaxRSS: `15683024K`

## Successful Run
- Dataset: `vidore/arxivqa_test_subsampled`
- Split: `test`
- Query/image count: `500`
- Model: `vidore/colqwen2.5-v0.1`
- Top-k: `5`
- Output dir: `/scratch/lmalveau/vidore_arxivqa_colqwen_runs/20260625T012935Z`
- Output files: `dataset_preview.jsonl`, `image_embeddings.pt`,
  `query_embeddings.pt`, `metrics.json`, `rankings.jsonl`

## Metrics

| Metric | Value |
| --- | ---: |
| `nDCG@5` | `0.09762220732374786` |
| `hit_at_1` | `0.058` |
| `hit_at_3` | `0.106` |
| `hit_at_5` | `0.136` |
| `mrr_at_5` | `0.08506666666666665` |

On the 0-100 scale used in the paper tables, this is approximately `9.76`
`nDCG@5`.

## Published Reference
`colpali.md` Table 7 reports ArxivQA `nDCG@5` values of `54.4` for the ColPali
reference row and `56.2` for the ColQwen2 row. The SOL run is therefore much
lower than the expected benchmark behavior.

## Follow-Up
Before treating the ViDoRe calibration as valid, inspect whether the local
script's gold mapping assumption is correct for `vidore/arxivqa_test_subsampled`.
The script currently treats row `i` as the positive image for query `i`; if the
dataset schema uses explicit positive ids or a different query-image mapping,
the metric calculation will under-report true retrieval quality.
