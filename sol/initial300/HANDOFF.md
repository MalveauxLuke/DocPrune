# Initial 300: ColQwen and MinerU on SOL

Owner authorized 2026-09-15: process the frozen 300-question cohort with ColQwen
and MinerU, using the browser shell, rsync, scratch storage, and economical
compatible hardware. This authorizes preparation, smoke, and production
preprocessing. It does not authorize teacher/answerer runs or selector training.

## Binding identities

- Checkout: `/home/lmalveau/DocPrune`. Supply the exact tested Git commit as
  `DP300_COMMIT` on every submission. Launchers reject revision or scoped-code
  drift; `experiments/initial300/CODE_SHA256SUMS` seals Python/helper bytes.
- Scratch root: `/scratch/lmalveau/docprune-initial300/20260915-v1`.
- Input root: that root plus `/input`, with paths relative to the local
  `/Users/god/DocPrune/outputs/datasets-v1` root.
- Frozen manifest SHA256:
  `f1942130e3b585ede4474515089dda2e3df67bb78a53f8f3acf9cd8b5322a495`.
- 150 SlideVQA, 100 DUDE, 50 TAT-DQA questions; 300 document families;
  3,743 logical pages. Unique pixel-identical pages are encoded once, preserving
  every document/page alias. No sampling changes or gold-page insertion.
- Transfer only the selected materializations and cohort records (3,209 files,
  approximately 546 MiB), plus a SHA256 manifest. No credentials/model files.
- ColQwen environment:
  `/home/lmalveau/.conda/envs/docprune-colfeatures17/bin/python`;
  torch 2.6.0, transformers 4.53.3, colpali-engine 0.3.12.
- ColQwen base `vidore/colqwen2.5-base` at
  `92908120384b7a2110c5beda3ab29cbdb2c08e49`, adapter
  `vidore/colqwen2.5-v0.2` at `dcbe8d9cede518bce830488364ba0e40c873645b`.
  Reuse existing scratch caches specified in `environment.sh`.
- MinerU environment:
  `/scratch/lmalveau/docprune/tool-envs/random-coverage-attribution-v2/mineru/bin/python`;
  installed MinerU 3.0.9, mineru-vl-utils 0.2.8, torch 2.13.0,
  transformers 4.57.6. Existing pypdfium2 supplies CPU PDF rendering.
- MinerU model `opendatalab/MinerU2.5-Pro-2604-1.2B` at
  `d3f5e08d073c21466bbabe21c71bb1e9c2e595da`; existing scratch cache.
- Both stacks: BF16, SDPA, explicit CUDA device, local cached models only,
  `PYTHONNOUSERSITE=1`; no CPU offload, no environment mutation.

## Processing contract and efficiency

1. CPU preparation verifies all transferred checksums, renders DUDE at 200 DPI,
   preserves official released SlideVQA/TAT-DQA images, and seals a deduplicated
   page catalog. Checksums, rendering, and model work run on compute nodes.
2. ColQwen encodes each unique page and each question once. Batch size four;
   similar page aspect ratios are batched together. Save normalized projected
   128-dimensional embeddings, real nonpadding token identities, image positions,
   native spatial grids, and boxes. Do not store redundant pixel tensors or huge
   preprojection states. Preserve the existing exact 506-tensor adapter check.
3. MinerU runs `batch_layout_detect` in batches of four: region types, normalized
   boxes, order, angles and raw layouts, with headers/lists retained. Skip crop
   transcription because segmentation is the requested output. These outputs
   are explicitly layout-only, not full OCR or postprocessed middle JSON.
4. CPU combination scores every page within its question's supplied document,
   saves full ordered rankings, and materializes top-20 query/patch and region
   profiles using positive-area geometric overlap. Missing overlaps are explicit.
   Whole-document page caches support later different K without model reruns.

The two production engines are independent after CPU preparation. Each worker
loads only its own model, once. Smoke runs engines sequentially to measure them
without simultaneous model residency. Separate arrays permit different shard
sizes; concurrency and time are finalized from measured smoke throughput.

## Jobs and resources

All jobs use htc/public. Generic GPU request, with capability >=8 verified in
code; no H100/H200 restriction. Verify scheduler acceptance using `--test-only`.

| Stage | GPU | CPUs | RAM | Wall time |
|---|---:|---:|---:|---:|
| CPU catalog | 0 | 1 | 4 GiB | 20 min |
| Sequential GPU smoke | 1 | 2 | 24,000 MiB | 20 min |
| GPU worker | 1 | 2 | 24,000 MiB | measured smoke sets final time |
| CPU ranking/profiles | 0 | 2 | 4 GiB | 20 min |

The GPU RAM request follows the observed SOL admission floor; this is host RAM,
not VRAM. Two CPUs serve preprocessing and accelerator feeding. Smoke verifies
actual memory usage before scaling. Prior ColQwen17 used about 8.35 GiB GPU
memory and 8.03 GiB peak host RSS with larger outputs; those are historical
measurements, not a substitute for the new smoke.

Example submission from the allocated browser shell:

```bash
cd /home/lmalveau/DocPrune
export DP300_COMMIT=<tested-Git-commit>
mkdir -p /scratch/lmalveau/docprune-initial300/20260915-v1/logs
sbatch --export=ALL sol/initial300/prepare.sbatch
sbatch --dependency=afterok:<prepare-job> --export=ALL sol/initial300/smoke.sbatch
```

After smoke success, inspect resource accounting, adapter parity, all expected
artifacts and visually inspect layouts across all three sources. Generate a
validated admission receipt with `admit.py --root ... --job <smoke-job>
--visual-review passed`. Do not claim quality from valid JSON alone.

Submit worker arrays only after receipt validation, with `DP300_ENGINE`,
`DP300_SHARDS`, `DP300_SMOKE_RECEIPT`, and `DP300_COMMIT` explicitly exported.
Initial planning estimate: four ColQwen shards and eight MinerU shards, at most
2 active per engine (4 GPUs overall); revise using smoke throughput before
submission. Then submit `combine.sbatch` afterok both arrays. Record actual
job IDs, commit, resources and measurements in the processing findings.

## Outputs, validation and recovery

`catalog.json`, `prepare-completion.json`, `colqwen/{pages,queries,runs}`, and
`mineru/{pages,runs}` live in the scratch root, as do model contracts and logs.
`combined/` contains full rankings, top-20 region profiles and completion receipt.
The artifacts are an explicit preprocessing format; Stage 2 reader-token
partitioning/import validation remains a later integration step.

Existing completed page/query units are reused only under identical contracts.
Source/contract/hash mismatch stops execution. Orphan tensors are preserved for
inspection, never silently accepted or deleted. Retry failed/incomplete shards
only after diagnosis. No input deletion, no widening of cohort, no model/backend
switches to make a failed run appear valid. Implementation fixes may be committed
and retried with a new pin; preserve incompatible earlier outputs separately.

Do not modify other running jobs or their environments. The historical
17-question acquisition record is retained in `PRIOR_SOL_TASK_20260915.md`.

## Owner-approved minimal diagnostic plus MinerU smoke — 2026-09-15

Run `diagnose.sbatch` with the tested `DP300_COMMIT`: one generic GPU,
2 CPUs, 24000M host memory, 20 minutes. The owner explicitly narrowed the
investigation to the original four pages, exact embedding differences and
associated-question retrieval-score differences, followed by MinerU smoke.

ColQwen executes one four-page batch, four singleton page encodings, and one
associated-question encoding per page. It saves per-page measurements before
proceeding, in `diagnostics/batch-JOB_ID/`. No repeated, reordered, identical-
image, alternative-attention, or explicit-position experimental arms run.

The original 0.995 threshold is reported unchanged. Its failure does not stop
this diagnostic from reaching MinerU; this is not production admission. The
launcher executes MinerU's original 10-page smoke after the ColQwen process
exits, releasing its GPU memory first. It records both process exit codes and
returns failure if either process errors. ColQwen diagnostic success only
means measurements were collected, not that batching was approved. Preserve
all previous failed evidence and do not create a production admission from
this job's exit status alone. No full production arrays or teacher runs.
