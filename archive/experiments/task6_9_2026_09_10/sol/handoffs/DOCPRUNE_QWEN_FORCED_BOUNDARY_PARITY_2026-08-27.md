# Qwen forced-boundary all-kept parity handoff — 2026-08-27

## Authority and scope

Status: **COMPLETED — admitted by job `62265662`; no further submission
authorized.** This handoff
supersedes the non-runnable draft handoff
[`DOCPRUNE_QWEN_FORCED_BOUNDARY_PARITY_DRAFT.md`](DOCPRUNE_QWEN_FORCED_BOUNDARY_PARITY_DRAFT.md).

This authority permits exactly one non-array L40S job that runs only:

```text
tests/qwen2vl/test_model.py::test_real_model_forced_all_kept_boundaries_match_stock_without_download
```

It is a seven-boundary all-kept no-op admission gate at `B_input`, `B_0`,
`B_6`, `B_13`, `B_20`, `B_23`, and `B_26`. It is not a retrieval, evaluation,
quality, feature-building, indexing, or benchmark job.

## Immutable pins

| Item | Exact value |
|---|---|
| Runtime | `/home/lmalveau/DocPrune-forced-boundary-runtime-36cb771` |
| Runtime commit | `36cb771db3af91ee7a0b76803ec099b12dba31e4` |
| Runtime launcher | `/home/lmalveau/DocPrune-forced-boundary-runtime-36cb771/examples/sbatch/33_docprune_qwen_forced_boundary_parity.sbatch` |
| Launcher SHA-256 | `bda211dc9c2ef958842d1323ef481fd94b2a9a1af0be344c88d3eae00b099054` |
| Environment | `/home/lmalveau/mamba-envs/docprune-sol` |
| Cached model | `/scratch/lmalveau/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac` |
| Model revision | `eed13092ef92e448dd6875b2a00151bd3f7db0ac` |
| Fixed probe | `/scratch/lmalveau/docprune/datasets/m3docvqa/probe/000ebcad8623837eb1384098a8c31d40-page-1.png` |
| Probe SHA-256 | `3ae33f3bc9064df02ef3535a3e7ed6a2b2dafbbc25519cea38db78b393ee19d4` |
| Fresh root | `/scratch/lmalveau/docprune/qwen-forced-boundary-parity-36cb771-v1` |
| Output directory | `/scratch/lmalveau/docprune/qwen-forced-boundary-parity-36cb771-v1/result` |
| Boundary artifact | `/scratch/lmalveau/docprune/qwen-forced-boundary-parity-36cb771-v1/result/forced-boundary-parity.json` |
| Resources | one L40S GPU, 4 CPUs, 64 GiB host memory, 20 minutes |

The runtime was verified clean and at the listed exact commit. The launcher
and probe hashes match the listed SHA-256 values, the cached model directory
and environment Python exist, and the fresh root was absent at handoff seal.

## Exact submission command

This exact command was submitted once as job `62265662` on 2026-08-27. It
submitted the launcher from the sealed clean runtime rather than the working
repository and created the otherwise-absent root immediately before submission;
the launcher then requires a fresh `result` directory and a fresh artifact path
and never replaces either. Do not submit it again.

```bash
ROOT=/scratch/lmalveau/docprune/qwen-forced-boundary-parity-36cb771-v1; test ! -e "$ROOT" && mkdir -- "$ROOT" && sbatch --parsable --chdir="$ROOT" --constraint=l40s --export=ALL,RUNTIME_DIR=/home/lmalveau/DocPrune-forced-boundary-runtime-36cb771,RUNTIME_COMMIT=36cb771db3af91ee7a0b76803ec099b12dba31e4,OUTPUT_DIR=/scratch/lmalveau/docprune/qwen-forced-boundary-parity-36cb771-v1/result,FORCED_BOUNDARY_ARTIFACT=/scratch/lmalveau/docprune/qwen-forced-boundary-parity-36cb771-v1/result/forced-boundary-parity.json,MODEL_PATH=/scratch/lmalveau/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac,PROBE_PAGE=/scratch/lmalveau/docprune/datasets/m3docvqa/probe/000ebcad8623837eb1384098a8c31d40-page-1.png,PYTHON=/home/lmalveau/mamba-envs/docprune-sol/bin/python /home/lmalveau/DocPrune-forced-boundary-runtime-36cb771/examples/sbatch/33_docprune_qwen_forced_boundary_parity.sbatch
```

Do not alter the launcher, its SHA-256, the runtime, model revision, probe,
resource request, constraint, test node, output root, or artifact name. Do not
submit an array or a second job. Record the returned job ID in the experiment
log before inspecting outcomes.

## Preconditions and forbidden operations

The launcher fails closed unless the runtime is clean, its `HEAD` exactly
matches the lower-case 40-hex pin, the output and artifact paths are fresh,
and the model/probe/environment paths exist. It enforces offline Hugging Face
and Transformers operation, while the test itself loads the processor and
model with `local_files_only=True`.

This job must not retrieve pages or model files, load any retrieval or global
index, rebuild page features, evaluate M3DocVQA questions, run any other test
node, submit further jobs, or modify source, tests, plans, or draft files.

## Success, evidence, and recovery

### Completed admission — 2026-08-27

Job `62265662` completed `0:0` on `scg017` in `00:00:26`. It used NVIDIA L40S
(46,068 MiB, driver `595.71.05`) and passed the sole pytest node in `19.88s`:
`1 passed`, with three non-failing generation-configuration warnings. JUnit
records `tests=1`, `failures=0`, `errors=0`, and `skips=0`.

The result root is
`/scratch/lmalveau/docprune/qwen-forced-boundary-parity-36cb771-v1/result`.
Every entry in `artifacts.sha256` verifies. Whole-file SHA-256 values are
`artifacts.sha256` `649bde404af0cc63a30436ba884a990ef8a0abfa99dc0377e5b0c69745528b1a`,
`forced-boundary-parity.json` `32748c311eaa993aa88128e87393fa36b5a2a10d5776cd871ae9e32e8a9d8302`,
`junit.xml` `6e037e92570f2069b6eeacbf35828a94aa64a4605b69f3f7a72404816502edc9`,
and `pytest.log`/Slurm output
`58f1314ce14880520ff2a43748dc863e5a0847db02d10dddf084a90c130478b7`.

The schema-1 boundary JSON has the exact ordered boundaries `B_input`, `B_0`,
`B_6`, `B_13`, `B_20`, `B_23`, and `B_26`. Every record is
`physical_delete`/`forced`, has visual population/requested/achieved budget
`2508`, retains all visual IDs, passes all six identity/parity verdicts, and
has M-RoPE digest
`1af5ac1559e09c2c1bc83fc52ce1c0fe28cb381f7b737998468bbec23b4ab077`.
Maximum and mean absolute first-step logit differences were `0.125` and
`0.012369606643915176`; the declared relative-tolerance verdict passed.

Success requires exit `0` and one passing pytest node. The node must emit the
fresh boundary JSON with one accepted all-kept record for each of the seven
boundaries, exact generated-suffix parity through EOS or the 128-token cap,
and first-step logits within `rtol=0.02`, `atol=0.07`. The launcher records
GPU name/memory/driver, runtime commit evidence, probe hash, JUnit XML, and
pytest log, then writes `artifacts.sha256` over those files plus the boundary
JSON.

On any preflight, test, artifact, or hash failure, preserve the root exactly as
written, record the job ID and available hashes/failure in the experiment log,
and stop. This handoff grants no retry, no replacement root, no tolerance
change, and no Task 4 or quality experiment.
