# Overlap-First V1 MiniVGent Proof of Concept

## Status and authority

- Design decision: owner approved the staged program on 2026-08-14 and
  narrowed its first comparison on 2026-08-14.
- Active POC: Qwen pairwise R0/R1 versus MiniVGent M0/M1 on one shared frozen
  V1 candidate universe.
- Owner review: approved for implementation-plan preparation in
  [`OWNER_REVIEW.md`](OWNER_REVIEW.md); this does not activate SOL.
- Later experiment: HierDoc H0/H1 and the A1 autoregressive-ID control are
  retained for a separately gated answer-sufficiency experiment. They are not
  active POC arms and are not part of the first SOL implementation handoff.
- Execution status: no SOL stage is active until the written specification and
  POC implementation plan are reviewed and the workspace gate is closed.
- Activation rule: the POC is authorized as a whole, but stages activate one
  at a time through `sol/CURRENT_SOL_TASK.md`.
- Immutable source: `/home/lmalveau/overlap_first_document_corpus/v1`.

This directory separates durable architecture from experiment scheduling. It
consolidates the former V1 segment-reranker baseline and MiniVGent proposal
without collapsing their model architectures.

## Initial POC decision

```text
immutable V1
  -> frozen DeepSeek-OCR-2 semantic candidates and answer-anchor view
  -> R0: stock pairwise Qwen
  -> R1: audited-hard-negative-tuned pairwise Qwen
  -> M0: shared-page MiniVGent without candidate interaction
  -> M1: shared-page MiniVGent with candidate interaction
```

The decisive first questions are:

1. Does hard-negative adaptation improve independent Qwen candidate scoring?
2. Does one question-conditioned full-page Qwen memory improve over repeated
   pairwise candidate crops?
3. Does off-diagonal candidate interaction improve over the same shared-page
   MiniVGent with candidate interaction disabled?

Qwen R0/R1 is mandatory because MiniVGent freezes that same Qwen reranker
backbone. Jina may be a separately registered optional side arm, but Jina
Reranker M0 or Jina Embeddings v4 cannot replace the matched Qwen control.

## Why HierDoc is later

HierDoc is valuable, but it is not the clean first-POC baseline. It introduces
an 8B generative backbone, alias-overlaid pages, autoregressive XML actions,
reflection, and label-based GRPO. V1 supplies answer anchors rather than
complete sufficient evidence sets, so its precision/F1 reward can punish
useful unlabeled context.

HierDoc therefore moves to the later answer-sufficiency experiment, after the
project has evidence sets or frozen-answerer sufficiency supervision capable of
supporting compact-set precision. A1 remains paired with HierDoc because it is
needed to separate action-space benefits from model scale and GRPO. The full
paper analysis, H0/H1 design, exact reward, alias contract, and A1 control are
preserved in architecture files and the later-experiment plan.

## Claim boundary

The active POC labels accepted answer-bearing anchors. It can support claims
about:

- candidate-oracle coverage;
- supplied-page answer-anchor ranking/localization;
- independent versus shared-page scoring;
- duplicate/distractor competition and redundancy; and
- the incremental effect of off-diagonal MiniVGent candidate interaction.

It cannot support claims about complete evidence, answer sufficiency,
necessity, valid empty-set behavior, genuine multi-hop reasoning, full-document
page retrieval, structured-ID policy quality, or answer-feedback optimization.

V1 mutation, DeepSeek-OCR-2 fine-tuning, page-policy training, synthetic
multi-hop, answer-feedback RL, split redesign, conflict adjudication, and
complete-evidence claims remain outside the active POC.

## Document map

### Active POC architectures

- [`architectures/candidate_and_supervision.md`](architectures/candidate_and_supervision.md):
  V1, DeepSeek candidates, eligibility, anchor labels, negative semantics, and
  common input boundaries.
- [`architectures/qwen_pairwise_reranker.md`](architectures/qwen_pairwise_reranker.md):
  R0/R1 scoring, hard-negative mining, training, and matched comparison.
- [`architectures/minivgent.md`](architectures/minivgent.md): exact Qwen memory,
  ROI, candidate tensors, M0/M1 decoder, losses, parameters, and runtime.

### Later answer-sufficiency architectures

- [`architectures/hierdoc_region_policy.md`](architectures/hierdoc_region_policy.md):
  preserved H0/H1 paper adaptation, aliases, action grammar, reward, and risks;
  not an active POC arm.
- [`architectures/qwen_autoregressive_control.md`](architectures/qwen_autoregressive_control.md):
  preserved A1 control required for a later HierDoc comparison.

### Experiment planning

- [`experiment/program.md`](experiment/program.md): active R0/R1/M0/M1
  questions, contrasts, stages, and interpretation.
- [`experiment/artifact_contract.md`](experiment/artifact_contract.md):
  immutable inputs, locks, outputs, resume, and recovery.
- [`experiment/metrics_and_statistics.md`](experiment/metrics_and_statistics.md):
  candidate-oracle, ranking, MiniVGent, slice, cost, and uncertainty metrics.
- [`experiment/gates.md`](experiment/gates.md): active POC stop and promotion
  gates.
- [`experiment/stages/`](experiment/stages/): active Stages 00-05.
- [`experiment/later_answer_sufficiency/README.md`](experiment/later_answer_sufficiency/README.md):
  deferred HierDoc/A1 and answer-sufficiency sequence.
- [`../../superpowers/plans/2026-08-14-overlap-v1-minivgent-poc.md`](../../superpowers/plans/2026-08-14-overlap-v1-minivgent-poc.md):
  consolidated staged implementation plan awaiting owner review.

### Preservation audit

- [`SOURCE_COVERAGE.md`](SOURCE_COVERAGE.md) records source hashes and maps the
  former baseline, MiniVGent, HierDoc, Qwen-readiness, and implementation-plan
  material to active or later destinations.

### Owner review

- [`OWNER_REVIEW.md`](OWNER_REVIEW.md) records the reviewed decisions,
  promotion standard, remaining implementation-plan locks, and execution
  boundary.

## Authority order

With the owner review recorded:

1. this README controls active-versus-later scope;
2. architecture files control data/model semantics;
3. active experiment files control POC stages, gates, artifacts, and metrics;
4. the later answer-sufficiency file controls only deferred planning;
5. the reviewed [POC implementation plan](../../superpowers/plans/2026-08-14-overlap-v1-minivgent-poc.md)
   controls file-level construction;
6. `sol/CURRENT_SOL_TASK.md` controls the one active SOL stage; and
7. older source documents remain provenance.

A SOL task may narrow but not broaden its active stage. No active POC task may
download, implement, train, or evaluate HierDoc/A1 merely because their later
architecture files exist.

## Active stage model

```text
POC authorized
  -> Stage 00 candidates/oracle
  -> Stage 01 Qwen R0/R1 and audited negatives
  -> Stage 02 MiniVGent implementation/preflight
  -> Stage 03 one-seed M0/M1 screen
  -> Stage 04 three-seed confirmatory comparison
  -> Stage 05 optional locked holdout
```

Each stage emits a create-once hashed completion report. The control plane
reviews the stage gate and explicitly activates the next stage; completion does
not submit another job automatically.

## First executable boundary

The first binding handoff will activate only Stage 00. Before it exists, the
workspace registry must record an isolated checkout and branch, clean
local/remote commit parity, scratch root, candidate-builder config, create-once
outputs, and recovery authority.

No model download, inference, mining, optimization, HierDoc/A1 work, or holdout
evaluation is authorized by these design files alone.
