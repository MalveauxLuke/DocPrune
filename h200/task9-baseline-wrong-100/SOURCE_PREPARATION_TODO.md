# Remaining source/SOL preparation

The source/SOL preparation is complete. Keep all generated data outside Git.
H200 may prepare both isolated environments while the sealed CPU-input bundle
transfers, but it must wait for complete checksum verification before any
preprocessing or GPU work.

1. **[Complete] Seal fixed inputs.** Run `seal_task9_confirmation_inputs.py` against the
   sealed cohort and exact cached top-4 pages/features. Require exactly 100 QIDs,
   400 pages, `retrieval_run: false`, and `global_index_loaded: false`.
   Canonical root:
   `/scratch/lmalveau/docprune/task9-baseline-wrong100-inputs-c0c9bee-v1`.
2. **[Complete] Package the CPU inputs.** The bundle contains the unchanged
   cohort, selected rows, 400 PNGs, 253 unique PDFs, 253 unique feature shards,
   fixed fixture/QID inputs, and provenance metadata. It excludes the global
   index and all generated MinerU/geometry/mapping output.
3. **[Complete] Seal transfer provenance.** The 1,014 raw files total
   2,826,385,204 bytes and verify against `MANIFEST.sha256`, whose SHA-256 is
   `d5eb76ff0e0382d487387e31d9a28cde37ae1de8020644a111ec659d2f1f5bb5`.
4. **[In progress] Transfer through the Mac bridge.** H200 verifies every byte,
   then owns relocation, pinned MinerU, geometry, mappings, CPU validation,
   experiment smoke, production, and aggregation as ordered in `HANDOFF.md`.
