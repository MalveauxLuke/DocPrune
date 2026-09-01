# Current Task

## Scope

Implement a faithful, training-free reproduction of DocPrune for document QA,
grounded in the CVPR 2026 paper and supplement.

## State

Task 9 one-question development is complete. Historical job `62323129`
admitted the original 64+32 B13 masks and failed its then-active dual-target
gate. The later paired 256+64 diagnostics are also complete: B13 job `62423463`
and corrected input job `62424211` admitted all 320 masks; input preflight
attempt `62423876` created no output. Generated-response analysis is sealed at
`/scratch/lmalveau/docprune/task9-paired-256-diagnostics-c49abb5-v2/analysis.json`.
Accepted-answer analysis is sealed at
`/scratch/lmalveau/docprune/task9-paired-256-accepted-answer-0d40fad-v1/analysis.json`.

The accepted-answer B13 result has LDS `0.91484` and held-out RMSE `0.19368`
versus constant `0.99428`; coefficient-refit Spearman mean/min is
`0.66975`/`0.57412`, while selected-set Jaccard mean/min is `0.70779`/`0.65`.
The user approved bypassing the old `0.8` exact-set identity gate and proceeding
with a controlled answer-conditioned oracle pilot whose stability is judged by
budget-local fidelity and actual budgeted-set outcomes. B13 and `B_input` now
remain historical diagnostics; the active pilot uses each question's native
dynamic layer and budget.

The approved sequence now begins with a preliminary stratified-random
48-question pilot drawn from the authenticated 245-question BTP+QTP/no-CTP
pool: 24 baseline-correct and 24 baseline-wrong questions sampled randomly
within stratum. It is sealed at
`/scratch/lmalveau/docprune/task9-preliminary-random48-v1/cohort.json` with file
SHA-256 `123607a6a1226b4e3436f43cb82d45e64e8a3008e9ab6deefd7526efeabd0273`.
For each question, native aggregate-threshold DocPrune first and independently
freezes its crossing layer `l*_q`, exact budget `M_q`, and retained token set.
ContextCite and regional random use that layer and the closest attainable
whole-region cost at or below `M_q`. The preliminary pilot is complete. The
corrected implementation was committed at `2a66d79`; validator-fix runtime
`90f27d7ed8b99ad10f1a5fe405c131127456ae5d` produced the admitted results in
array `62464099`, with only six timed-out questions retried by array `62466113`.
The previously approved enriched 48-question developmental
mechanism panel follows: 16 uniformly sampled eligible questions, 16 traceable
distractor errors, 8 high-ambiguity correct questions, and 8 clean controls.
Both use 256 fit masks, 32
global and 32 native-budget-local holdouts, per-question dynamic physical
deletion, native DocPrune, gold-conditioned
attention, robust/canonical accepted-answer ContextCite, reverse ContextCite,
unpruned, and a traceable-only audited constraint. FastV and new random or
coverage pruning families are excluded. The preliminary pilot alone includes
one matched region-size-aware random comparator required for its same-action-
space interpretation. The existing 1,213-question Task 6 random-versus-
DocPrune evidence remains separate and is not rerun.
The 2026-08-31 non-executable preparation handoff is preserved as historical;
the amended experiment plan/log, this file, and runtime commit `90f27d7` are
current authority.
All unique accepted Task 6–9 Git work is now consolidated on
`codex/task9-analysis-cpu-20260828`: Task 6 merge `689432e`, Task 7 merge
`c5b212f`, and Task 8 merge `c38cd9c`.

The evaluation measurement fix and durable HTC shard pipeline are sealed at runtime
`4e2473bdbbc2e4eca0e92c30d4a0633044501ccf`. The validated schema-5 indexes are promoted without
altering their bytes into fresh attempt `/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2`.
The reusable paired top-4 256-question checkpoint completed as arrays `62008122` (DocPrune) and
`62008123` (all-kept). After inspecting those paired results, the user approved the full top-4
continuation. Original arrays `62030328` (DocPrune) and `62030329` (all-kept) cover exact remaining
shard IDs 4–38; the completed checkpoint shards remain final inputs and will not be recomputed.
DocPrune shards 4–16 and all-kept shards 4 and 6 completed valid. On 2026-08-24, original DocPrune tasks
17–38 and all-kept task 5 failed in 6–9 seconds at the clean-check preflight because generated,
untracked M3DocRAG `__pycache__` files made the pinned checkout dirty; no model was loaded and no
question was evaluated. The bytecode was preserved at `/tmp/m3docrag-pycache-20260823-2316`, the
checkout clean-check passes again, and authorized exact retries are `62041375` (DocPrune 17–38)
and `62041383` (all-kept 5). Original all-kept task 6 then passed preflight and completed valid as
job `62041373` with 64/64 questions. All-kept shards 7–10 later completed valid, bringing that
mode to 10/39 validated shards. Redundant shard-17 probe `62046530` remains held.

The user approved replacing only unfinished A100-80 work with mixed-hardware quality-only HTC
shards. Old pending records `62030329`, `62041375`, and `62041383` were canceled at zero additional
runtime after shard 10 validated. Supplemental control
`/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/broad-gpu-control.json` binds clean control
checkout `/home/lmalveau/DocPrune-broad-control-fdb5e91` at
`fdb5e918622a422cc90fbd6e62a610719ee49974`. Replacement arrays `62068296` (all-kept shards
5 and 11–38) and `62068302` (DocPrune shards 17–38) run at most eight independent tasks per array
on reviewed A100/H100/L40 GPUs with at least 39,000 MiB. Quality can be merged across hardware;
efficiency aggregation is explicitly excluded.

A quick stage-localization diagnostic was first sealed under
`/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/diagnostics/stage64-v1`.
It deterministically selects 64 questions, without using answer outcomes, from all 245 single-hop
questions currently complete in both modes. Runtime `2abcc22ea2f2d9d60849da5f89c37a7b16d5d9da`
adds BTP-only and BTP+QTP diagnostic factories. Initial four-by-16 arrays `62063320` and
`62063380` had no estimated start and were canceled at zero runtime with no results. Replacement
arrays `62065503` and `62065508` failed at preflight without evaluating questions because their
nominal runtime worktree had been advanced away from sealed commit `2abcc22`. Preserve that
advanced worktree unchanged. Retry arrays `62071272` and `62071274` loaded both models but failed
before their first question because diagnostic stages passed a non-finite CTP-disable threshold
to a controller that correctly rejects non-finite values. Commit
`607fc38e23198864705db62084fcd91fd2f234da` replaces that sentinel with the existing finite `1e9`
disable value and adds a controller-level regression test. Corrected diagnostic root
`/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/diagnostics/stage64-v2` preserves the exact
v1 question selection and reference. Two-by-32 HTC arrays `62072829` (BTP only) and `62072828`
(BTP+QTP) use clean detached runtime `/home/lmalveau/DocPrune-stage64-runtime-607fc38` and request
any supported 40GB-or-larger A100/H100/L40 GPU with 45-minute limits. They must validate retrieved
page IDs and order against the existing paired reference. They do not replace the full benchmark.
Arrays `62072828` completed both BTP+QTP shards and `62072829_0` completed BTP shard 0 with exact
reference retrieval. `62072829_1` completed 32 answers on an H100 but correctly failed the final
fidelity guard because 9/32 ordered page lists drifted. Preserve it but exclude it from the
controlled comparison. Isolated recovery root
`/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/diagnostics/stage64-v3-btp1-a10040`
binds the exact same shard-1 QIDs to A100-40GB-only recovery job `62074548_1`.

Recovery job `62074548_1` completed valid with exact reference retrieval. Across the controlled
64 questions, F1 is 41.0625 all-kept, 39.265625 BTP-only, 39.5625 BTP+QTP, and 39.625 full
DocPrune. The endpoint loss is -1.4375 F1, but every adjacent-stage bootstrap interval includes
zero, so this quick diagnostic is inconclusive. The approved expansion is sealed at
`/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/diagnostics/stage245-v1-incremental`.
It reuses plan shards 0-3 (the completed 64) and evaluates only the remaining 181 questions in
plan shards 4-15. Arrays `62075807` (BTP only) and `62075816` (BTP+QTP) each contain 12 independent
A100-40GB-only HTC jobs, at most 16 questions per job, using clean runtime
`/home/lmalveau/DocPrune-stage245-runtime-a2bd8f2` at
`a2bd8f27d6e38009daf0898926abb6787ad7d216`. Every completed shard must exactly match the sealed
245-question retrieved-page reference; H100 output is excluded from this controlled comparison.

Both diagnostic arrays completed 12/12 at exit 0. The sealed 245-question analysis is
`stage245-analysis.json` in that root. F1 follows 44.8612 all-kept -> 43.1143 BTP-only ->
43.6327 BTP+QTP -> 41.3347 full DocPrune. CTP loses 2.2980 F1 points with paired 95% interval
[-4.2980, -0.6286]; BTP and QTP intervals include zero.

The complete top-4 evaluation is finished at 39/39 valid shards and 2,441/2,441 rows in each
mode. Sealed analysis
`/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/analysis/top4-paired-quality.json` reports
37.7911 all-kept versus 36.7603 DocPrune F1. Single-hop is 46.4497 versus 44.8303; multi-hop is
24.8827 versus 24.7296. Ordered retrieved-page identities match on 2,126 questions; within that
controlled stratum the paired delta is -1.1980 F1 with interval [-2.3471, -0.0640]. Full
DocPrune drops 81.72% of original visual tokens after CTP versus the paper's 74% decoder drop.

Canonical quality-only merge packaging remains unpublished. Jobs `62086624`/`62086625` (32 GB)
and `62087740`/`62087864` (64 GB) failed only by OOM and published no output. The merger invokes
deep index validation for every already-validated shard, materializing 23.9 GB embeddings, a
2.8 GB JSON token map, and a 23.5 GB FAISS index. Do not request more memory blindly; correct the
quality merge path to authenticate saved successful shard validations without redundant full-index
materialization.

## Historical execution records (immutable, non-promotable)

Historical diagnostic attempt-2 at `/scratch/lmalveau/docprune/benchmark-6c19bfc/attempt-2/`: evaluation array `61830411` was canceled before
work, and six schema-4 indexes `61830405`–`61830410` completed `0:0` but
cannot be promoted. Preserve all scratch artifacts. Scheduling-only attempt-1
(gate `61883512`, indexes `61883881`–`61883886`, eval `61883887`, compare
`61883888`) was canceled before work at `00:00:00`; preserve it unchanged. The
old-runtime production graph `61968793`, `61968794`–`61968797`, `61968799`–`61968800`,
`61968821`, and `61968823` was canceled at `2026-08-21 17:50:21` with no nodes
and elapsed 0. Scheduling probes `61969352` and `61969614` were also canceled
with no node/elapsed 0. Attempts 3–7 are immutable failed or canceled L40/A100
probes: `61970394`, `61972695`, `61973090`, `61974092`, and `61974173`.
Preserve every listed root and ID unchanged. The active root is the fresh
`/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2`, using immutable runtime checkout
`/home/lmalveau/DocPrune-runtime-4e2473b` and the sharded control checkout/handoff below.

## Exact authority pins

```text
environment: /home/lmalveau/mamba-envs/docprune-sol
PDF tools: /home/lmalveau/mamba-envs/m3docvqa-acquisition
corpus: /scratch/lmalveau/docprune/datasets/m3docvqa
HF cache: /scratch/lmalveau/hf_cache; Hub cache: /scratch/lmalveau/hf_cache/hub
Qwen: Qwen/Qwen2-VL-7B-Instruct@eed13092ef92e448dd6875b2a00151bd3f7db0ac
ColPali: vidore/colpali-v1.2@961b51745de3e9adb3468ac5c9ccca0ac626c217
ColPali backbone: vidore/colpaligemma-3b-pt-448-base@30ab955d073de4a91dc5a288e8c97226647e3e5a
M3DocRAG: /home/lmalveau/src/m3docrag-benchmark-29e6ac2 @ 29e6ac2294d6b87075a1d45b8a8df175b214248a
runtime: /home/lmalveau/DocPrune-runtime-4e2473b @ 4e2473bdbbc2e4eca0e92c30d4a0633044501ccf
control: sealed full SHA in /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/control.json
attempt root: /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2
```

## Binding design

[`../docs/superpowers/specs/2026-08-20-docprune-paper-fidelity-correction-design.md`](../docs/superpowers/specs/2026-08-20-docprune-paper-fidelity-correction-design.md)

## Next action

Prepare the already-planned enriched pilot: seal its outcome-blind 16/16/8/8
cohort and implement the remaining gold-attention, robust/canonical/reverse
ContextCite, and audited traceable arms. Do not submit a method holdout. The
preliminary unified JSON is
`/scratch/lmalveau/docprune/task9-preliminary-dynamic48-90f27d7-unified-v1/analysis.json`
(internal SHA-256 `985a837b2094b5a925730eeb4b9bf07c594344f9021c4a718c49c11aa655a8c5`).
