# Stage 02: MiniVGent Implementation and Systems Preflight

## Activation scope

Stage 02 implements and verifies M0/M1 on a 128-512-question stratified
preflight view. It may run synthetic tests and a tiny real diagnostic overfit,
but no scientific one-seed screen. It authorizes no HierDoc or A1 work.

## Inputs

- passed Stages 00-01 reports;
- frozen candidate, eligibility, negative, R0/R1, and validation hashes;
- exact MiniVGent/Qwen architecture files;
- approved model/cache/environment/run roots; and
- exact stage commit.

## Procedure

1. Measure candidate p50/p90/p95/max, member counts, OCR token estimates,
   unknown types, overlaps, and resource implications before any cap.
2. Implement frozen config and typed tensors.
3. Reproduce the official Qwen prompt and scalar score path.
4. Implement selective layer-memory capture and all-hidden-state parity tests.
5. Implement row-major visual grid and actual-member ROI pooling.
6. Implement the exact candidate encoder and two-block M0/M1 decoder.
7. Implement stable OR/ranking/verified-negative losses.
8. Implement backbone freezing, parameter asserts, added-weight checkpoints,
   strict resume, and online frozen-Qwen trainer/evaluator plumbing.
9. Pass all synthetic/fake-model tests before loading Qwen.
10. On real Qwen, pass score/tap parity, token/grid, ROI overlays, M0 isolation,
    M1 equivariance, gradients, parameter counts, resource fit, checkpoint
    round trip, and 16-question overfit.

## Outputs

- exact model/environment/config locks;
- source-verified and real-model parity reports;
- visual-grid/ROI overlay audit;
- parameter, memory, latency, token, and candidate profiles;
- synthetic/unit/integration test results;
- tiny diagnostic checkpoint/overfit evidence outside Git; and
- Stage 02 completion report.

## Pass and stop

Pass every Stage 02 gate in `gates.md`. Any mismatch, orientation error,
pooled gap, backbone gradient, parameter discrepancy, checkpoint failure, or
OOM preserves evidence and stops the screen.

## Next-stage handoff

On pass, the control plane may activate Stage 03. Stage 02 does not run the
scientific M0/M1 screen or any later answer-sufficiency model.
