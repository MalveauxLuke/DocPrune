# SOL sharded acquisition — owner approved 2026-09-14

This handoff supersedes the smoke-only execution scope in HANDOFF.md.
The owner approved accepting efficient-attention numerical differences,
using the current all-content baseline, and submitting 17 questions / 3 arms
as a sharded SOL job. No separate smoke receipt is required: each question
runs the retained integration checks before revealing any arm observations.

## Experiment

Array task 1 through 17 selects Q01 through Q17, respectively. Each task loads
one pinned Qwen2.5-VL-7B instance and executes R (random), A (prepared Col-style),
and adaptive, with 32 observations each and exactly 5016/10032 visual tokens.
The three arms share one question baseline, hardware and physical score cache,
but only see scores for masks each independently requests. Total: 1632 logical
observations; duplicate requested masks reuse a physical measurement.
No fresh retrieval, rendering, feature extraction or modification of sealed inputs.

The all-content baseline is measured with efficient vision SDPA on that shard
and saved as baseline.json. Historical likelihoods are preserved as provenance
and their differences are reported, but historical numerical equality is no
longer an admission requirement. Adaptive decisions and all summary deltas use
the new baseline. The fixed G targets and original wrong-answer S target stay
unchanged. Exact original generated-answer/EOS checks and masked cached-versus-
legacy parity at 1e-4 remain mandatory; a failure stops only that shard.
The model and input/processor identities, original visual positions, exact
50% budget, finite scores and immutable resume records are still enforced.

## Runtime and resources

Read ../AGENTS.md and ../../docs/SOL_INSTRUCTIONS.md. Use the existing checkout
/home/lmalveau/DocPrune, main, pinned by ACQUISITION_CODE_COMMIT. Use existing
/home/lmalveau/mamba-envs/docprune-acquisition-sol/bin/python and environment.sh.
Inputs, model, all caches and outputs remain under
/scratch/lmalveau/docprune-adaptive-acquisition/20260914-smoke01.
Qwen snapshot cc594898137f460bfe9f0759e9844b3ce807cfb5, offline, pinned dependencies.
No environment installation or download is needed. Login nodes are for light
inspection, Git and submission; all package validation/inference runs in jobs.

Submit score.sbatch with the verified commit. Array 1-17%4 limits concurrency
to four GPUs; every shard has one GPU, 2 CPUs, 24000 MiB host RAM and 20 minutes.
Whole-GPU families a30/a100_40/a100_80/l40/l40s/h100 are eligible. The prior Q12
diagnostic used 19.4 GiB efficient GPU allocation and 10.9 GiB host RAM, finishing
both backends in 2m47s. The 20-minute limit allows 96 mask observations plus model
load and guards; this is an estimate, not a measured all-question runtime.
Question-local matching prevents between-device differences from confounding
arms. Device/model/runtime identities and peaks are recorded per shard.

## Outputs and recovery

Logs: score-ARRAY_ID_TASK_ID.log. Results: score-ARRAY_ID/shard-TASK_ID,
containing execution.json, QNN/baseline.json, identity.json, arm journals,
physical measurements and summary.json. Shards have distinct writer locks.
A complete run has 17 summaries with R=32, A=32, adaptive=32 each.
Decoded candidate review remains a separate assessment after acquisition.
Preserve diagnostic/smoke outputs and unrelated jobs. Retry only unfinished
shards, specifying the original ACQUISITION_SCORE_RUN directory. Resume requires
matching code/runtime identities and a repeated baseline within 1e-4; never
silently mix hardware or recomputed scores with an existing question history.
