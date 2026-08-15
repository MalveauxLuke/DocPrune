# ViDoRe ArxivQA Custom ColQwen2 Summary

## Verdict
CUSTOM COLQWEN2-V1.0 GOOD

The custom script works when using the official evaluator's known-good model,
`vidore/colqwen2-v1.0`, with `model_arch=colqwen2`. Its metrics closely match
the official evaluator run, so the earlier bad custom result points to the
ColQwen2.5 path/model compatibility rather than the custom script's row-index
gold mapping.

## Job
- Job id: `57144736`
- Job name: `vidore-arxivqa-custom-cq2`
- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:06:01`
- Batch MaxRSS: `16329716K`

## Run Inspected
- Output directory: `/scratch/lmalveau/vidore_arxivqa_custom_colqwen2_runs/20260625T031858Z`
- Slurm stdout: `logs/vidore_arxivqa_custom_colqwen2_57144736.out`
- Slurm stderr: `logs/vidore_arxivqa_custom_colqwen2_57144736.err`
- Metric file: `/scratch/lmalveau/vidore_arxivqa_custom_colqwen2_runs/20260625T031858Z/metrics.json`
- Rankings file: `/scratch/lmalveau/vidore_arxivqa_custom_colqwen2_runs/20260625T031858Z/rankings.jsonl`
- Dataset preview: `/scratch/lmalveau/vidore_arxivqa_custom_colqwen2_runs/20260625T031858Z/dataset_preview.jsonl`

## Configuration
- Model arch: `colqwen2`
- Model name: `vidore/colqwen2-v1.0`
- Dataset: `vidore/arxivqa_test_subsampled`
- Split: `test`
- Top-k: `5`

## Metrics

| Metric | Value |
| --- | ---: |
| `ndcg_at_5` | `0.8770630290361083` |
| `hit_at_1` | `0.818` |
| `hit_at_3` | `0.896` |
| `hit_at_5` | `0.928` |
| `mrr_at_5` | `0.8600999999999998` |

## Gold Coverage From Rankings
- Ranking rows: `500`
- Candidate-count distribution: `5 candidates` for `500` rows
- Gold@1: `409 / 500` (`0.818`)
- Gold@3: `448 / 500` (`0.896`)
- Gold@5: `464 / 500` (`0.928`)

Gold rank distribution:

| Rank | Count |
| ---: | ---: |
| 1 | 409 |
| 2 | 27 |
| 3 | 12 |
| 4 | 7 |
| 5 | 9 |

Integrity checks:
- Rows with invalid rank sequence: `0`
- Rows with scores out of descending order: `0`
- Rows with duplicate candidates: `0`
- Rows with mismatched gold flags: `0`

## Log Notes
No traceback, runtime error, missing-key warning, unexpected-key warning, or
scoring warning was found in the completed run logs.

Warnings observed:
- `TRANSFORMERS_CACHE` is deprecated; use `HF_HOME`.
- `torch_dtype` is deprecated; use `dtype`.
- `Qwen2VLImageProcessor` is loaded as a fast processor by default.

These warnings did not stop evaluation.

## Comparison
The custom ColQwen2 result (`ndcg_at_5 = 0.87706`) matches the official
evaluator's good result (`ndcg_at_5 = 0.87359`) and is far from the earlier bad
custom ColQwen2.5 result (`ndcg_at_5 = 0.09762`).

## Conclusion
Custom `colqwen2-v1.0` good => the ColQwen2.5 path is suspect. The custom
script can produce expected ArxivQA performance when model architecture and
checkpoint are aligned.
