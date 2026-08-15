# ViDoRe ArxivQA Official Evaluator Summary

## Verdict
OFFICIAL EVALUATOR NORMAL

The completed official ViDoRe evaluator run reproduces expected ArxivQA
performance. This strongly indicates the earlier low custom-script result is
from the custom evaluation path, not from the SOL environment, checkpoint, or
model loading.

## Job
- Job id: `57138474`
- Job name: `vidore-arxivqa-official`
- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:06:10`
- Batch MaxRSS: `36635332K`

## Run Inspected
- Output directory: `/scratch/lmalveau/vidore_arxivqa_official_runs/20260625T023910Z`
- Slurm stdout: `logs/vidore_arxivqa_official_57138474.out`
- Slurm stderr: `logs/vidore_arxivqa_official_57138474.err`
- Help file: `/scratch/lmalveau/vidore_arxivqa_official_runs/20260625T023910Z/evaluate_retriever_help.txt`

## Configuration
- Model class: `colqwen2`
- Model name: `vidore/colqwen2-v1.0`
- Dataset: `vidore/arxivqa_test_subsampled`
- Dataset format: `qa`
- Split: `test`
- Environment: `/home/lmalveau/mamba-envs/colqwen25`
- Official evaluator version in metrics: `5.0.0`

The installed official evaluator registry includes `colqwen2`, but no separate
`colqwen2.5` class. The CLI help describes `--model-class` but does not list
the supported class names.

## Metrics
Metric files:
- `/scratch/lmalveau/vidore_arxivqa_official_runs/20260625T023910Z/outputs/colqwen2_vidore_colqwen2-v1.0_metrics.json`
- `/scratch/lmalveau/vidore_arxivqa_official_runs/20260625T023910Z/outputs/colqwen2_vidore_colqwen2-v1.0/vidore_arxivqa_test_subsampled_metrics.json`

| Metric | Value |
| --- | ---: |
| `ndcg_at_5` | `0.87359` |
| `recall_at_1` | `0.814` |
| `recall_at_3` | `0.892` |
| `recall_at_5` | `0.924` |
| `mrr_at_5` | `0.8585333333333333` |

The official evaluator reports `nDCG@5 on vidore/arxivqa_test_subsampled:
0.87359`, which is in the expected ArxivQA ColPali/ColQwen range and not near
the bad custom-script result (`0.09762`).

## Log Notes
No traceback, runtime error, missing-key warning, unexpected-key warning, CLI
argument warning, or failed-job signature was present in the successful run.

Warnings observed:
- `TRANSFORMERS_CACHE` is deprecated; use `HF_HOME`.
- `torch_dtype` is deprecated; use `dtype`.
- `Qwen2VLImageProcessor` is loaded as a fast processor by default.

These warnings did not stop evaluation.

## Conclusion
Official evaluator normal => the custom script is wrong. The most likely issue
remains the custom script's gold mapping/evaluation assumption, especially
treating row `i` as the positive image for query `i`.
