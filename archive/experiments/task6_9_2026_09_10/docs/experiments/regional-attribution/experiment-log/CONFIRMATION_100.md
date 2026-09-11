# 100-question baseline-wrong confirmation log

## 2026-09-02 — Cohort sealed

- Eligible baseline-wrong questions: `731` across `724` support components.
- Selected: `100`, one per component; no fallback or distractor enrichment.
- Retrieval run: false.
- Cohort artifact:
  `/home/lmalveau/docprune-data/h200-artifacts/task9-baseline-wrong100-confirmation-v1/cohort.json`
- Internal cohort SHA-256:
  `0bdfdde29b568f54ea541453abeca196cfb49f4b6630f084f80e6c21f158514e`

## 2026-09-02 — Fixed inputs and transfer bundle

- Fixed inputs: 100 questions and 400 cached page slots, with no retrieval or
  global-index loading.
- Source fixture SHA-256:
  `9b54c41787c0ccc34205c8e3fc23ca50701b93dc1c74289e64b57740c3ed9a71`.
- Transfer subset: 1,014 files and 2,826,385,204 bytes.
- Transfer-manifest SHA-256:
  `d5eb76ff0e0382d487387e31d9a28cde37ae1de8020644a111ec659d2f1f5bb5`.
- H200 owns MinerU, geometry, mapping, validation, experiment execution, retry,
  and aggregation after authenticating the transferred bytes.

## 2026-09-02 — H200 environment and relocation

- Separate DocPrune and MinerU environments passed targeted imports and
  `pip check`; the stacks remain separate because their validated dependencies
  conflict.
- The transferred manifest re-authenticated successfully before relocation.
- CPU relocation found 100 selected records, 400 PNGs, 253 PDFs, and 253
  feature shards with no retrieval or global index.
- No scientific confirmation outcome is recorded yet.

Append subsequent preprocessing, smoke, production, retry, and aggregation
events here. Exact commands remain in the
[H200 handoff](../../../../h200/task9-baseline-wrong-100/HANDOFF.md).
