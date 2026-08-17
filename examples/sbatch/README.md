# SOL SBATCH Examples

These project-neutral examples teach the repository's cluster conventions.
They do not authorize a run. Before submission, an approved SOL handoff must
name the exact checkout, commit, environment, inputs, scratch root, resources,
command, outputs, and recovery authority.

## Examples

| File | Purpose | Source pattern |
|---|---|---|
| [`00_lightwork_setup.sbatch`](00_lightwork_setup.sbatch) | Lightweight environment and repository inspection | Distilled from `render_mmlongbench_ocr2_313.sbatch` |
| [`01_gpu_smoke.sbatch`](01_gpu_smoke.sbatch) | Short one-GPU hardware/runtime smoke test | Distilled from `run_mmlongbench_ocr2_deep_parse_smoke.sbatch` |
| [`02_single_gpu_run.sbatch`](02_single_gpu_run.sbatch) | Normal single-A100 research job shell | Distilled from `run_segment_evidence_stage1_baselines.sbatch` |
| [`03_job_array.sbatch`](03_job_array.sbatch) | Bounded four-shard array pattern | Extends the retained safety, provenance, and scratch conventions |
| [`10_docprune_smoke.sbatch`](10_docprune_smoke.sbatch) | Pinned environment, GPU imports, tests, and config smoke | Active DocPrune reproduction |
| [`11_docprune_m3docvqa.sbatch`](11_docprune_m3docvqa.sbatch) | Gated top-1/top-2/top-4 evaluation runner | Active DocPrune reproduction |
| [`20_m3docvqa_download_array.sbatch`](20_m3docvqa_download_array.sbatch) | Pinned M3DocVQA dev PDF acquisition shards | Staged acquisition/probe handoff |

The named source wrappers are preserved below
`../../sol/archive/source-repo/archive/jobs/`.

## Before submitting

1. Read [`../../docs/SOL_INSTRUCTIONS.md`](../../docs/SOL_INSTRUCTIONS.md).
2. Inspect current partitions and resources with `sinfo`.
3. Create or verify environments from a compute allocation, never a login
   node.
4. Keep large caches and outputs under `/scratch/$USER`.
5. Replace the default inspection command only after a binding handoff exists.

The examples write Slurm stdout/stderr in the submission directory so the
path exists before Slurm opens it. Move or collect important logs according to
the approved handoff.
