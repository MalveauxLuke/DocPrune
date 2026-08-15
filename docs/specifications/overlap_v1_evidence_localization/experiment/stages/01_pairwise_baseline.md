# Stage 01: Pairwise Qwen Baseline and Audited Adaptation

## Activation scope

Stage 01 builds and measures mandatory R0/R1. It also creates the audited
training-only negative artifacts inherited by M0/M1. It cannot alter Stage 00
candidates or open internal test/holdout.

## Inputs

- passed Stage 00 completion report;
- frozen candidates, relations, eligible/excluded questions, and oracle hashes;
- Qwen primary source/model/environment proposal from the pairwise architecture;
- approved model/cache/run roots; and
- exact stage commit and commands.

## Procedure

1. Implement shared ranking evaluation and document-clustered comparison with
   hand-computable tests.
2. Implement the immutable Qwen adapter/model lock and official score parity.
3. Run an eight-pair GPU smoke and reject non-finite or gold-leaking payloads.
4. Score R0 on complete train and validation candidate manifests. Do not create
   internal-test predictions in this stage.
5. Mine training-only hard/ordinary non-anchors with all removal guards.
6. Complete the deterministic stratified audit of at least 200 mined hard
   candidates and report valid/false-negative/uncertain rates.
7. Freeze the R1 training manifest and config; train from the R0 lock with
   validation-only checkpoint selection.
8. Score R1 on the byte-identical validation manifest.
9. Produce validation R0/R1 metrics, slices, costs, paired deltas, and
   bootstrap interval. Stage 04 creates both arms' first internal-test
   predictions after the whole program is frozen.

Optional Jina scoring may occur only under a separately registered config and
cannot delay, replace, or modify Qwen R0/R1.

## Outputs

All Stage 01 locks, train/validation predictions, negative groups/audit, R1
training evidence, checkpoints, validation comparison, and completion report
in the artifact contract.

## Pass and stop

The stage passes when integrity, audit, leakage, training, and matched-input
gates pass. R1 may fail to outperform R0 without blocking Stage 02; that is a
scientific result. Inability to certify verified negatives blocks all later
training that consumes them.

## Next-stage handoff

On pass, the control plane locks R0/R1 artifacts and may activate Stage 02.
Stage 01 does not install/train MiniVGent and performs no HierDoc/A1 work.
