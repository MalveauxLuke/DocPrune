# Current SOL Task

## State

- Prepared, not yet launched.
- Objective: acquire pinned DeepSeek-OCR-2 output for every unique valid gold
  page used by the answerable questions in the full MMLongBench-Doc dataset.
- Binding contract:
  `task_spec/mmlongbench_gold_pages_deepseek_ocr2.md`.
- Dataset root:
  `/scratch/lmalveau/agenticdocai/data/MMLongBench-Doc`.
- Result branch: `codex/mmlongbench-gold-pages-ocr2`.
- No job ID or run ID exists yet.

## Next action

1. Read `../AGENTS.md`, `../docs/SOL_INSTRUCTIONS.md`, `AGENTS.md`, and the
   binding contract.
2. Inspect the dataset layout and the repository's completed MMLongBench OCR
   pipeline before changing code.
3. Build and validate the deterministic complete gold-page inventory, freeze
   its observed counts and hashes, and add focused tests.
4. Render and run a deterministic smoke set through the pinned OCR-2 path.
5. Submit the complete sharded OCR run only after the smoke and provenance
   gates pass.

## Hard boundary

SOL performs OCR acquisition and Git-safe packaging only. Do not run semantic
segmentation, hierarchical routing, Qwen reranking, answer generation,
evaluation, or website work.

## Blockers

- The authoritative dataset table, page-number convention, and final inventory
  counts must be discovered and frozen from the supplied SOL dataset before
  inference. Do not substitute the retired academic-subset counts.
