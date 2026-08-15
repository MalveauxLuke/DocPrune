# Overlap-First V1 Evidence Localization

## Role

Active owner-approved single-hop answer-anchor POC over immutable V1. It
unifies the former segment-reranker baseline and MiniVGent experiment under one
candidate, supervision, evaluation, and artifact contract while preserving
separate model architectures.

## Arms

- R0/R1: stock and hard-negative-tuned pairwise Qwen controls.
- M0/M1: shared-page MiniVGent without/with candidate interaction.

HierDoc H0/H1 and the matched A1 autoregressive-ID control are retained under
the later answer-sufficiency experiment. They are not active POC arms.

## Stage sequence

```text
00 candidates/oracle
  -> 01 R0/R1 and audited negatives
  -> 02 MiniVGent systems preflight
  -> 03 one-seed R0/R1/M0/M1 screen
  -> 04 three-seed confirmatory run
  -> 05 optional locked holdout
```

The whole program is authorized; only the stage named by
`sol/CURRENT_SOL_TASK.md` is executable.

## Boundaries

- Source: `/home/lmalveau/overlap_first_document_corpus/v1`.
- One frozen candidate revision; all filtering is derived and audited.
- Answer-anchor labels do not establish complete evidence.
- Qwen R0/R1 remains mandatory; Jina is optional.
- Holdout is sealed until Stage 05.
- HierDoc/A1, answer-sufficiency training, and page routing require a separate
  later-stage authorization.
- No V1 mutation, OCR-2 fine-tuning, multi-page routing, synthetic multi-hop,
  answer-feedback RL, conflict adjudication, or split redesign.

## Authority and recovery

- Full design:
  `docs/specifications/overlap_v1_evidence_localization/README.md`
- Architecture details:
  `docs/specifications/overlap_v1_evidence_localization/architectures/`
- Stages, metrics, artifacts, and gates:
  `docs/specifications/overlap_v1_evidence_localization/experiment/`
- Current state: `agent-context/CURRENT_TASK.md`
- Workspace gate: `docs/EXPERIMENT_WORKSPACES.md`

Planning is active. No SOL stage is active until the written design and unified
implementation plan are reviewed and the workspace/handoff gate is closed.
