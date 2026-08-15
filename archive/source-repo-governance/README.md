# COLPALI Research Central Command

This repository is the control plane for COLPALI document-understanding
research. It records research direction, canonical specifications, current
authority, experiment locations, compact provenance, final findings, and
organized historical records. Large datasets, caches, generated overlays, and
other heavy outputs belong under named scratch roots rather than Git.
It does not execute experiments from the control checkout.

The overlap-first document corpus V1 is complete. The active follow-on is a
staged answer-anchor POC comparing the pairwise Qwen baseline with MiniVGent.
HierDoc and the matched autoregressive control are retained for a later
answer-sufficiency test. Execution remains blocked until written-spec and
implementation-plan review plus a dedicated workspace, branch, scratch root,
and Stage 00 SOL handoff.

## Current authority: overlap-first V1 evidence localization

The program is authorized to:

1. freeze one DeepSeek-OCR-2 semantic-segment candidate revision;
2. derive an audited reranker view containing only questions with an
   answer-anchor-covering segment;
3. benchmark mandatory stock/tuned Qwen pairwise controls on
   `(segment, original_query)`;
4. mine and audit training-only hard non-anchor candidates;
5. fine-tune the same selected reranker; and
6. verify and screen MiniVGent M0/M1 against R0/R1; and
7. run a frozen three-seed POC comparison and optional locked holdout.

The entire POC is approved, but SOL activates one stage at a time. HierDoc
H0/H1 and A1 require later answer-sufficiency authorization. V1 remains
immutable. DeepSeek-OCR2 fine-tuning, page-policy training, synthetic
multi-hop, answer-feedback RL, conflict adjudication, complete-evidence claims,
and split redesign remain outside scope.

## Sources of truth

- [Unified program specification](docs/specifications/overlap_v1_evidence_localization/README.md)
- [Architecture files](docs/specifications/overlap_v1_evidence_localization/architectures/)
- [Experiment stages and gates](docs/specifications/overlap_v1_evidence_localization/experiment/)
- [Source-preservation audit](docs/specifications/overlap_v1_evidence_localization/SOURCE_COVERAGE.md)
- [Current agent task](agent-context/CURRENT_TASK.md)
- [Current SOL task](sol/CURRENT_SOL_TASK.md)
- [SOL operating rules](docs/SOL_INSTRUCTIONS.md)

## Repository and workspace roles

The control checkout remains this repository on `main`. No experiment checkout
or scratch root has been approved for the POC yet; see the explicit
`destination-required` gate in `docs/EXPERIMENT_WORKSPACES.md`.

## Active dataset boundary

The immutable source is:

```text
/home/lmalveau/overlap_first_document_corpus/v1
```

It contains 12,856 documents, 66,551 canonical questions, 65,787 usable
questions, and 23,043 page positions. DocVQA and DUDE retain document-grouped
80/10/10 splits. InfographicsVQA remains isolated in its holdout.

## Start here

- [Repository navigation](docs/NAVIGATION.md)
- [Active and paused projects](docs/ACTIVE_PROJECTS.md)
- [Experiment workspaces and destination gates](docs/EXPERIMENT_WORKSPACES.md)
- [Branch roles](docs/BRANCHES.md)
- [Agent context index](agent-context/INDEX.md)
- [Research architecture registry](agent-context/research/architecture_registry/README.md)
- [Archive policy](archive/README.md)
- [Project archive index](archive/projects/README.md)

## Fast routing verification

```bash
git diff --check
python -m pytest -q tests/test_documentation_contract.py tests/evidence_dino/test_handoff.py --tb=short
```

## Completed and retained research

The overlap-first acquisition and packaging work is complete and archived.
Evidence-DINO-Units, hierarchical evidence routing, the MMLongBench
segmentation lab, the gold-page reranker, and the earlier segment-reranker
corpus remain retained. The `main-project` query-planner worktree remains
separately active and protected.
