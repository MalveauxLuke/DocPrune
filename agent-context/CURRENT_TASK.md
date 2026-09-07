# Current task

## Current owner request — 2026-09-07

Package the 40-candidate evidence-reviewed correction corpus and a baseline-first
input-versus-intermediate deletion-depth comparison for H200. See the scoped
[correction-depth handoff](../h200/correction-depth/HANDOFF.md). Each depth gets
its own 256-mask gold-support oracle; margin is secondary. Packaging and CPU
validation are authorized here; no GPU execution is part of this task.

Latest owner clarification: use the existing H200 DocPrune checkout, with new
artifacts only under its ignored local-data folder. Segment the admitted corpus
with MinerU before baselines; capture geometry and publish mappings afterward.
Do not create another source checkout.

Packaging is complete locally. The [delivery directory](/home/lmalveau/DocPrune/outputs/correction-depth-delivery-2026-09-07/README.md)
contains source/input archives, checksums, validation and missing-asset details.
All 40 candidate definitions are included; 36 have complete copied assets.
The residual addition reuses the same G/S mask scores for separate self fits
and coefficient differences. Transfer/assembly and one-case H200 execution
remain pending; no GPU results are claimed.

The 600 questions are already on H200, with segmentation pending in the latest
owner update. Their separate preparation and the 100-question confirmation
remain untouched. Reuse authenticated transferred assets read-only where they
exist. The older track descriptions below describe those separate activities,
not instructions to launch them during correction-corpus work.

## Goal

Complete Task 9's confirmatory and feasibility experiments without changing the frozen retrieval, model, prompt, decoding, physical-deletion, or cohort contracts.

## Established evidence

- The Task 6–9 implementation and histories are consolidated on `main`.
- The 48-question development pilot supports continuing the adapted regional ContextCite diagnostic: it preserved 24/24 baseline-correct answers and rescued 4/24 baseline-wrong answers relative to native DocPrune.
- The 192-mask reduction failed the frozen functional gate; 256 fitting masks remain the supported configuration.
- Prefix/boundary-state caching is already implemented. Additional cost reduction requires batching or a separately validated adaptive design, not another basic cache rewrite.

See [the consolidated results](../docs/results/README.md) for exact values and artifact paths.

## Active Task 9 tracks

### 1. Baseline-wrong confirmation

- Cohort: 100 new baseline-wrong questions, one per support-document component.
- Inputs: only the exact cached top-four pages and persisted features used by the existing pipeline.
- Method: native DocPrune and matched regional ContextCite evaluation with 256 fitting masks and no extra 64-mask fidelity holdout.
- Compute: H200 under the CoRAL machine policy.
- Authority: [baseline-wrong handoff](TASK9_H200_BASELINE_WRONG100_HANDOFF_2026-09-02.md) and [H200 execution handoff](../h200/task9-baseline-wrong-100/HANDOFF.md).

### 2. Shared-probe feasibility study

- Cohort: sealed, document-disjoint 600-question cohort kept separate from the 48- and 100-question cohorts.
- Target: one fixed physical-deletion boundary, `B13`.
- Probes: independent pre-answer decoder-block snapshots `B0..B13`, followed by a controlled compact local-trajectory model.
- Initial scope excludes model families 0 and 1 as recorded in the research-review addendum.
- GPU preparation and execution occur on H200; SOL owns source implementation, CPU validation, packaging, and result analysis.
- Authority: [canonical experiment plan](../docs/experiments/regional-attribution/EXPERIMENT_PLAN.md), [SOL handoff](TASK9_SHARED_PROBE_SOL_HANDOFF_2026-09-03.md), and [H200 handoff](../h200/task9-shared-probe/HANDOFF.md).

## Nonnegotiable boundaries

- Do not run retrieval or load the global retrieval index.
- Do not mix the 48-question development pilot, 100-question confirmation, or 600-question shared-probe cohort.
- Do not describe regional ContextCite as vanilla ContextCite, a token-level oracle, or a deployable method.
- Do not replace 256 masks with 192 or 64 based on the development cohort.
- Do not launch a GPU job without the relevant machine handoff and explicit user approval.
- Record every admitted or rejected run in the Task 9 experiment log.

## Immediate next action

Synchronize the preserved H200 execution branch, validate its transferred inputs, and continue the two H200 tracks from their machine handoffs. Repository cleanup does not authorize either experiment to launch.
