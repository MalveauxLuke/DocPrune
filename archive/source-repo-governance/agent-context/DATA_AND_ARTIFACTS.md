# Data and Artifacts

This central-command checkout retains specifications, routing, small
provenance records, tests, and final findings. Heavy or generated experiment
artifacts belong in the approved external workspace and scratch root recorded
in `docs/EXPERIMENT_WORKSPACES.md`.

## Current physical root

```text
Immutable V1: /home/lmalveau/overlap_first_document_corpus/v1
```

The POC code checkout, scratch root, and environment are not yet
approved. Do not reuse older Evidence-DINO, BoundingDocs, or MMLongBench paths.

## V1 contents

The package contains:

```text
documents.jsonl
questions.jsonl
pages.jsonl
ocr.jsonl
excluded_questions.jsonl
manifest.json
README.md
```

Images and DeepSeek-OCR-2 payloads remain in the verified external locations
recorded by the package and are referenced by path and SHA-256.

## Git-safe experiment material

- candidate, Qwen, and MiniVGent configuration schemas;
- small source/model/candidate revision locks;
- checksums and manifest summaries;
- candidate-oracle and exclusion reports;
- hard-negative audit summaries;
- metric summaries and paired-delta reports;
- focused implementation and tests; and
- current task, branch, job, and recovery identifiers.

## External or scratch-only material

- rendered segment crops and masked page views;
- full candidate and relation tables;
- R0/R1/M0/M1 prediction payloads;
- model downloads, caches, optimizer state, and checkpoints;
- full training logs and profiling traces;
- Slurm stdout/stderr; and
- temporary review images.

Scratch is temporary. A scientific result is not complete until its exact
source hashes, candidate revision, model lock, counts, metrics, exclusions,
and failure disposition exist in a Git-safe manifest or report.

## Split boundary

- `train`: hard-negative mining and optimizer updates.
- `validation`: model choice, threshold fitting, early stopping.
- `test`: one frozen opening in Stage 04 after validation decisions are locked.
- `infographicsvqa_holdout`: sealed until optional Stage 05.

Never write derived candidates, labels, exclusions, or scores back into V1.
