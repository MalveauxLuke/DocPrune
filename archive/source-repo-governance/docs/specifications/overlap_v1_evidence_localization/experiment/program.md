# Active MiniVGent POC Program

## Goal

Test MiniVGent evidence localization against the matched Qwen pairwise
reranker before adding generative region policies or answer-sufficiency
optimization.

All active arms use immutable V1, one frozen candidate revision, the original
question, common accepted answer-anchor alternatives, audited negative
semantics, and common held-out manifests.

## Active arms

| Arm | Architecture | Training | Role |
|---|---|---|---|
| R0 | Qwen pairwise candidate scorer | released weights | stock independent baseline |
| R1 | same Qwen scorer | audited train hard non-anchors | tuned pairwise baseline |
| M0 | frozen full-page Qwen memory plus diagonal-only candidate decoder | supervised anchor loss | shared-page control |
| M1 | same as M0 with bidirectional candidate interaction | identical supervised loss | MiniVGent treatment |

H0/H1/A1 are not active arms. They remain documented under
`later_answer_sufficiency/`.

## Registered contrasts

```text
delta_pairwise = R1 Recall@1 - R0 Recall@1

delta_shared_memory = M0 Recall@1 - R1 Recall@1

delta_interaction = M1 Recall@1 - M0 Recall@1
```

- `delta_pairwise` measures audited hard-negative task adaptation.
- `delta_shared_memory` measures the operational value of a single
  question-conditioned full-page memory versus repeated candidate crops. It is
  not a pure causal ablation because the visual representation differs.
- `delta_interaction` is the cleanest architecture ablation: M0/M1 match
  candidate features, initialization, rows, objective, width, depth, and
  schedule; only off-diagonal candidate communication differs.

## Hypotheses

Primary:

> M1 improves answer-anchor Recall@1 over M0 by at least 2.0 absolute points in
> the three-seed confirmatory run, with positive direction in all three seeds
> and a document-clustered 95% confidence interval excluding zero.

Secondary:

- R1 improves over R0.
- M0 improves over R1 without exceeding the binding 2.0-point noninferiority
  margin on the matched deployment metric.
- M1 gain persists when the candidate OCR does not contain the accepted
  answer string.

Failure is reportable. It does not authorize test-driven expansion to HierDoc,
new candidate types, Qwen unfreezing, or answer feedback.

## Included scope

- DocVQA/DUDE V1 train, validation, and one locked internal-test opening;
- one supplied OCR-backed page per question;
- frozen DeepSeek semantic candidates and answer-anchor OR alternatives;
- candidate-oracle and eligibility/exclusion audit;
- mandatory Qwen R0/R1 and audited training negatives;
- exact two-block M0/M1 implementation and gated four-block promotion;
- common ranking, slice, cost, and systems metrics; and
- one optional locked InfographicsVQA robustness evaluation.

## Excluded scope

- HierDoc H0/H1 and A1 in the active POC;
- V1 mutation or DeepSeek-OCR-2 fine-tuning;
- complete evidence, answer sufficiency, necessity, or no-evidence labels;
- page routing, synthetic multi-hop, answer-feedback search/distillation/RL;
- downstream answer-model training;
- Qwen LoRA/backbone unfreezing;
- split redesign/conflict adjudication; and
- holdout-driven model choice.

## Stages

| Stage | Name | Result |
|---:|---|---|
| 00 | candidates and oracle | common frozen universe and candidate ceiling |
| 01 | pairwise baseline | R0/R1 plus audited negative dependency |
| 02 | MiniVGent systems preflight | verified Qwen memory, ROI, tensors, losses, and resource fit |
| 03 | one-seed M0/M1 screen | bounded architecture and depth decision |
| 04 | confirmatory run | three-seed full-train comparison and one internal-test opening |
| 05 | locked holdout | optional frozen InfographicsVQA robustness test |

Only the active stage in `sol/CURRENT_SOL_TASK.md` may execute. Later files do
not broaden an active POC task.

## Data progression

Stage 01 trains R1 on all eligible training rows after the negative audit.
Stage 03 uses a deterministic source- and document-balanced sample of up to
10,000 eligible training questions for M0/M1 screening. The earlier
15,000-20,000 estimate remains historical guidance, not a required count.
Stage 04 trains promoted M0/M1 on all eligible training rows for three seeds.

Validation controls checkpoint selection, score thresholds, MiniVGent depth,
and promotion. Internal test remains unopened until Stage 04 registration
freezes every POC choice. Holdout remains unopened until Stage 05.

## Score-to-set boundary

Primary comparisons use rankings and require no threshold. If selected-set
views are reported, fit one deterministic rule on validation and freeze it
before internal test. Selected-set analysis remains answer-anchor analysis, not
answer sufficiency.

## Depth promotion

Stage 03 starts with two-block M0/M1. Promote four-block M1 only when two-block
M1 is stable, improves validation Recall@1 or MRR over M0 without material
regression in the other, and fits the approved allocation. Otherwise confirm
the two-block architecture. No six-block or 2,048-wide expansion is allowed.

## Confirmatory protocol

Before Stage 04, freeze candidate/view/input hashes, model revision, prompt,
processor, M0/M1 depth, loss, optimizer, schedules, thresholds, seed list, and
checkpoint rules. Train M0/M1 for three seeds, lock checkpoints, then create
the first internal-test predictions for R0/R1/M0/M1.

MiniVGent promotion requires:

- M1-minus-M0 Recall@1 at least +2.0 absolute points;
- positive direction in all three seeds;
- document-clustered 95% interval excluding zero;
- gain on answer-string-absent rows;
- M1 no more than 2.0 absolute Recall@1 points behind R1; and
- no failed permutation, ROI, manifest, checkpoint, parameter, or
  frozen-backbone verification.

The 2.0-point noninferiority margin is a binding project choice, not a paper
result.

## Failure interpretation

- Poor oracle: repair/refreeze candidates; do not score models.
- R1 does not beat R0: hard-negative adaptation failed; retain both controls.
- M0 beats R1: shared page memory helps.
- M1 beats M0: off-diagonal interaction helps single-hop competition or
  redundancy, not proven multi-evidence complementarity.
- M0 beats R1 but M1 does not beat M0: shared memory helps; interaction does
  not.
- M1 gain only on answer-string-present rows: treat as an OCR shortcut.
- M0/M1 do not beat R1: deploy the simpler tuned pairwise reranker.

## Later answer-sufficiency experiment

HierDoc becomes appropriate after supervision can score compact evidence sets
without treating useful unlabeled context as false. That later experiment will
pair H0/H1 with A1, common complete/sufficient-evidence sets, and a frozen
answerer or independently audited sufficiency evaluation. It is not an
automatic next POC stage.
