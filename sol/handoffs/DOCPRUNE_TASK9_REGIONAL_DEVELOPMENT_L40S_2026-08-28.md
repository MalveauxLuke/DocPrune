# Task 9 One-Question Regional Development L40S Handoff

Date: 2026-08-28
Status: submitted exactly once as HTC job `62323129` after explicit user
approval. The job was pending for priority at the first scheduler check. Do
not run `sbatch` again or submit an automatic retry.

## Objective and claim boundary

Run the first real Task 9 development feasibility experiment for QID
`e1e6ed53f9ad11813845088f4cf2f6b1` at `B_13`. One exact-L40S job scores the
canonical 64 fit masks (`0..63`) followed by 32 held-out masks (`64..95`) in
one shared-prefix scoring call. The scorer internally performs the additional
unpruned BTP+QTP/no-CTP generation needed to bind the secondary target.

This is privileged, answer-conditioned whole-region attribution adapted from
ContextCite machinery. It is not vanilla ContextCite, evidence ground truth, a
token oracle, or a deployable query-only selector. It is one-question
development feasibility, not a holdout or confirmatory experiment.

## Exact clean runtime

```text
runtime: /home/lmalveau/DocPrune-task9-development-runtime-7616b29
checkout: clean detached worktree
commit: 7616b29b4dc5ba33584a6e26371281be6886188f
environment: /home/lmalveau/mamba-envs/docprune-sol
M3DocRAG: /home/lmalveau/src/m3docrag-task6-clean-20260828
M3DocRAG commit: 29e6ac2294d6b87075a1d45b8a8df175b214248a
Qwen: Qwen/Qwen2-VL-7B-Instruct
Qwen revision: eed13092ef92e448dd6875b2a00151bd3f7db0ac
launcher: examples/sbatch/39_docprune_task9_regional_development.sbatch
launcher SHA-256: 963522965d13c530ab9f4a2ecc8c424882919cdbe00025e25f76ad57409753df
```

The launcher requires both Git checkouts to match these commits and remain
clean. It also hashes every fixed input before model loading.

## Frozen inputs

```text
QID: e1e6ed53f9ad11813845088f4cf2f6b1
boundary: B_13
fixture: /scratch/lmalveau/docprune/task6-fixed-page-gate-v1/fixture-stage64-top4-v2.json
fixture SHA-256: 32b3ddd6a1f608db509f002f9541dbc92317b59ac769f0b41fcb20bcc536652b
run config: /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/run-configs/docprune-top4-task6-clean-m3-v1.json
run config SHA-256: 2233621303ccdf531267bb5bd2a7670775e54f90d04dede4fb19f264e8cad502
fixed feature manifest: /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/indexes/docprune/top4/docprune/manifest.json
fixed feature manifest SHA-256: ffa5979b3bf157adefcc132b0438af295ddafb2243377db4cdb8fcb8eaafe5da
mapping: /scratch/lmalveau/docprune/task8-region-mapping-f3f5de7-v1/region-token-mapping.json
mapping file SHA-256: a1c986232259c4979b6c3e423f866b8ff107f1f430ed96feaf5062ce42d04ab6
mapping internal SHA-256: 993de7e7b9e1e0a728a3839c30821338a4e742abb43b8594133921362313857b
geometry count: 3586
geometry SHA-256: 47b32cf6156dcee41da7eab1686219c760dd73803a135534546c1a50519086e8
trace: 10032 -> 4336 -> 3586 visual tokens
```

The mapping has 95 nonempty intervention sources: 80 MinerU regions and 15
residual cells. The audited zero-token region remains provenance evidence but
is excluded from the 95-column regression design.

The later CPU stability calculation uses the already observed canonical L40S
aggregate-native requested budget `M=2689`, not a value selected from Task 9
outcomes. Its source is:

```text
/scratch/lmalveau/docprune/task6-smoke-8b02837-v2/l40s/run/results.jsonl
file SHA-256: a96bf92265701f193945719e9ecc73078ce4cd97e6bf97fa73fa72694d144e58
row identity: aggregate-native-threshold, B_14, requested/achieved budget 2689
```

## Resources and fresh output

```text
partition: htc
qos: public
GPU: exactly one NVIDIA L40S
CPU: 8
host RAM: 24 GiB
time limit: 00:10:00
requeue: disabled
shards: none
fresh job root: /scratch/lmalveau/docprune/task9-regional-development-7616b29-v1
terminal artifact: /scratch/lmalveau/docprune/task9-regional-development-7616b29-v1/output
later CPU analysis: /scratch/lmalveau/docprune/task9-regional-development-7616b29-v1/analysis.json
```

The root was confirmed absent while preparing this handoff. The admitted
four-mask smoke measured about 2.28 GiB host MaxRSS and 16.55 GiB peak GPU
allocation. The 24-GiB request is the HTC GPU floor. One job avoids repeated
model loading and caches the `B_13` decoder prefix once for all 96 branches.

## Submitted command — provenance only; do not repeat

Run exactly once from a login node:

```bash
test ! -e /scratch/lmalveau/docprune/task9-regional-development-7616b29-v1
mkdir /scratch/lmalveau/docprune/task9-regional-development-7616b29-v1
sbatch \
  --export=ALL,RUNTIME_DIR=/home/lmalveau/DocPrune-task9-development-runtime-7616b29,RUNTIME_COMMIT=7616b29b4dc5ba33584a6e26371281be6886188f,JOB_ROOT=/scratch/lmalveau/docprune/task9-regional-development-7616b29-v1 \
  /home/lmalveau/DocPrune-task9-development-runtime-7616b29/examples/sbatch/39_docprune_task9_regional_development.sbatch
```

This command returned `Submitted batch job 62323129` at
`2026-08-28T19:57:28`. Slurm recorded partition `htc`, QOS `public`, one node,
8 CPUs, 24 GiB, one GPU, feature `l40s&public`, a 10-minute limit, and
`Requeue=0`.

No array, shard, dependency, A100 portability job, or automatic retry is
authorized.

## Output and terminal admission

The runner publishes without replacement and writes
`completion-manifest.json` last. The terminal directory contains exactly:

```text
run-manifest.json
raw-result.json
primary-target.json
secondary-target.json
completion-manifest.json
```

The primary target is the maximum accepted-reference normalized full-sequence
log-likelihood. The secondary target is the single normalized full-sequence
likelihood of the exact unpruned generated non-EOS token IDs. Both datasets
derive from the same raw 96-row likelihood matrix and identical physical masks.

The launcher runs the outcome-blind terminal validator after publication. It
authenticates completion-manifest-last member bytes and internal hashes, the
clean runtime, fixed inputs, exact seed/vector order, both target identities,
generated IDs and terminal EOS, no-CTP generation trace, 96 physical cache
records, full-to-compact topology, Qwen M-RoPE shapes, the exact trace and
L40S identity, and the absence of retrieval/global-index execution. It
reconstructs both normalized outcome vectors from the raw per-sequence rows.

Do not inspect or analyze target values unless the terminal validator returns
`admitted-task9-regional-development`.

## Post-admission CPU analysis

Only after terminal admission, run directly on the lightwork CPU allocation:

```bash
PYTHONPATH=/home/lmalveau/DocPrune-task9-development-runtime-7616b29/src:/home/lmalveau/mamba-envs/docprune-sol/lib/python3.10/site-packages \
PYTHONNOUSERSITE=1 \
/scratch/lmalveau/docprune/tool-envs/random-coverage-attribution-v2/contextcite/bin/python \
  /home/lmalveau/DocPrune-task9-development-runtime-7616b29/examples/analyze_task9_regional_development.py \
  --root /scratch/lmalveau/docprune/task9-regional-development-7616b29-v1/output \
  --output /scratch/lmalveau/docprune/task9-regional-development-7616b29-v1/analysis.json \
  --runtime-commit 7616b29b4dc5ba33584a6e26371281be6886188f \
  --qid e1e6ed53f9ad11813845088f4cf2f6b1 \
  --boundary B_13 \
  --fixture-sha256 32b3ddd6a1f608db509f002f9541dbc92317b59ac769f0b41fcb20bcc536652b \
  --mapping-sha256 a1c986232259c4979b6c3e423f866b8ff107f1f430ed96feaf5062ce42d04ab6 \
  --mapping-internal-sha256 993de7e7b9e1e0a728a3839c30821338a4e742abb43b8594133921362313857b \
  --geometry-count 3586 \
  --geometry-sha256 47b32cf6156dcee41da7eab1686219c760dd73803a135534546c1a50519086e8 \
  --original-visual-tokens 10032 \
  --post-btp-visual-tokens 4336 \
  --post-qtp-visual-tokens 3586 \
  --decoder-layer-count 28 \
  --gpu-substring L40S \
  --requested-budget 2689
```

For each target, this reports the per-question held-out LDS/Spearman value,
RMSE versus the fit-target-mean constant, and five-refit coefficient/selection
stability. It intentionally produces no LDS confidence interval. The official
interval is deferred to a subsequently sealed multi-question development set
and support-component bootstrap.

## Failure and recovery limits

- Preserve every partial or failed root unchanged.
- A root without a valid terminal completion manifest is not admissible.
- Do not inspect partial target values to choose a recovery.
- Diagnose scheduler, runtime, input, model, schema, or validator failure first.
- Any retry requires a fresh root and a newly reviewed exact handoff.
- Do not switch GPU family, shard the masks, change seeds, target identities,
  mapping, boundary, pages, features, model, prompt, decoder, or budget.
- Do not submit the A100 portability probe or any large/method-holdout run.

After the one-question raw artifact and analysis are admitted, the next
pre-outcome scientific decision is to seal the number and identities of
additional development questions before viewing their top-versus-reverse
results. This handoff does not make that decision.
