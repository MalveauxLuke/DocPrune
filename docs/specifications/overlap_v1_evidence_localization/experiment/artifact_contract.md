# Artifact and Reproducibility Contract

## Principles

1. V1 is immutable.
2. Every derived artifact is create-once under an explicit revision/run root.
3. Every consumer verifies hashes before reading.
4. No stage overwrites a predecessor's output.
5. Git contains code, configs, small manifests, reports, and provenance—not
   images, datasets, weights, caches, predictions, environments, or Slurm logs.
6. A failed stage preserves failure evidence and does not activate its
   successor.

## Required locations before Stage 00

The workspace registry and SOL task must pin:

- local execution checkout and branch;
- SOL checkout path and branch;
- exact local and remote commit SHA;
- approved scratch root;
- persistent Git-safe report return path;
- Hugging Face cache/model storage roots;
- environment location;
- recovery owner and restart instructions; and
- the one active stage output root.

Login nodes perform Git, light inspection, and submission only. Environment
creation, downloads, hashing at scale, image work, inference, mining, and
training require a Slurm allocation. Code/small configs live under the SOL
home checkout; large active outputs live under `/scratch/$USER/...` and are not
treated as permanent archival storage.

## Program and stage locks

Every run manifest contains:

- `program_id` and program-spec hash;
- `stage_id` and stage-spec hash;
- Git commit and clean-worktree assertion;
- V1 manifest and source hashes;
- candidate revision and manifest hashes;
- eligibility/exclusion/relation/view hashes;
- model, source, license, environment, processor, prompt, and config locks;
- input artifact hashes;
- deterministic seeds and data-order hash;
- hardware, Slurm job, timestamps, commands, and exit state;
- output hashes and create-once root; and
- predecessor stage completion-report hash.

Mutable branch names, Hub branches, `latest` tags, unrecorded package upgrades,
or path-only identities are invalid locks.

## Stage 00 artifacts

- `candidates.jsonl`;
- `relations.jsonl`;
- `eligible_questions.jsonl`;
- `excluded_questions.jsonl`;
- `candidate_manifest.json`;
- `candidate_oracle_report.json`;
- deterministic render/crop hashes;
- candidate review overlays outside Git; and
- `stage_00_completion.json` plus Git-safe Markdown summary.

The candidate table contains no query-relative labels. Relations and
eligibility join candidates to questions under the exact revision.

## Stage 01 artifacts

- Qwen model/environment/prompt/processor lock;
- R0 train/validation prediction manifests; internal test remains unopened;
- ranking metrics and per-query rows;
- hard-negative group manifest;
- at-least-200-row audit sample and completed audit decisions;
- mining and false-negative report;
- R1 configs, logs, checkpoints, selected-checkpoint lock, and validation
  predictions;
- matched validation R0/R1 comparison and document-clustered interval; and
- `stage_01_completion.json` plus Git-safe report.

Checkpoints and predictions stay outside Git; small locks and reports return to
Git-safe paths.

## Stage 02 artifacts

- exact Qwen model/environment/prompt/processor lock;
- Qwen score/tap parity report;
- visual-grid and ROI overlay audit;
- M0/M1 synthetic and real preflight reports;
- parameter, memory, latency, token, and candidate-count profiles;
- checkpoint round-trip and 16-question overfit evidence; and
- `stage_02_completion.json`.

## Stages 03-05 artifacts

Every active model run preserves raw create-once predictions, scores, ranks,
metrics, costs, slices, failures, model/config/checkpoint hashes, and exact
input-view hashes.

Stage 03 additionally freezes the screen sample and architecture-decision
report. Stage 04 freezes the confirmatory registration before the first
internal-test predictions for any arm, including R0/R1, and preserves per-seed
plus aggregate results. Stage 05 records proof that every
model/prompt/threshold/config hash matches Stage 04.

## Prediction schema

Every candidate-score row includes:

- run/stage/model/config/candidate/view hashes;
- question, document, page, candidate, and split IDs;
- scalar logit/score and deterministic rank;
- relation class only in the evaluation join, not model-input payload; and
- token, pass, latency, memory, and rendering-cost fields where available.

Answers and gold boxes stay in the evaluation relation table, not serialized
model inputs.

## Resume and overwrite safety

- Resume keys are semantic identities such as `(question_id,candidate_id)` or
  `(run_id,question_id,pass)`—never line counts.
- A resume validates every upstream hash and rejects a changed config/model/
  candidate/view/environment.
- Shards merge only under identical locks and non-overlapping keys.
- Existing output files are never silently truncated or replaced.
- Checkpoints include optimizer/RNG state where training resume is supported.
- M0/M1 checkpoints store added weights only plus locks; Qwen weights are
  referenced by immutable model identity.

## Candidate/model/view change control

- Candidate-definition change -> new `candidate_revision`, Stage 00 restart,
  all downstream runs invalidated.
- Prompt/processor/score change -> new model/input config and matching control
  reruns.
- Loss/reward/threshold/hyperparameter change -> new registered config; never
  retrofit old predictions.
- Architecture width/depth/taps/features/mask change -> new named arm/config.

HierDoc/A1 aliases, actions, rewards, decoded outputs, and later sufficiency
artifacts are governed by `later_answer_sufficiency/README.md` and require a
separate artifact contract before execution. They are not active POC outputs.

## Completion report

Each `stage_NN_completion.json` records:

```text
status: complete | failed | stopped
gate_result: pass | fail | owner_review_required
inputs: exact hashes
outputs: exact hashes
checks: named results
metrics: stage-defined summary
failures: counts and paths
next_stage_eligible: boolean
recommended_next_action: factual, not self-authorizing
```

Only the control plane may use this evidence to activate a next SOL task.
