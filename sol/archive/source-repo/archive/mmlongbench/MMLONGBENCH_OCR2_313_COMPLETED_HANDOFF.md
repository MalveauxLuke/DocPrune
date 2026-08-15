# Current SOL Task

## Purpose

Run the 313-page MMLongBench-Doc DeepSeek-OCR-2 off-domain pilot. The binding
contract is `sol/task_spec/mmlongbench_deepseek_ocr2_segmentation_pilot.md`.

## Required Reading

1. `AGENTS.md`
2. `SOLinstrucitons.md`
3. `sol/AGENTS.md`
4. `sol/task_spec/mmlongbench_deepseek_ocr2_segmentation_pilot.md`

## Current State

- Prepared branch: `codex/mmlongbench-ocr2-313`
- Input: `pilot_data/mmlongbench_ocr2_unstructured_313/`
- Source scope: 10 PDFs, 313 rendered pages, every PDF at most 50 pages.
- Explicit user override: the 306 attempt-1 pages that passed validation are
  packaged as an incomplete exploratory subset; the seven excluded pages were
  not retried.
- Run: `/scratch/lmalveau/mmlongbench_ocr2_unstructured_313/runs/20260717T182300Z`
- Input validation and focused tests passed; render and one-page OCR smoke passed.
- Result bundle: `sol_results/mmlongbench_ocr2_unstructured_313/20260717T182300Z/`
- Bundle audit: 306 pages, 7 exclusions, 4,218 unique segments, checksums valid.
- SOL renders and runs OCR only. SOL must not build semantic sections or a webpage.

## Next Action

Push result commit `a59d47f` from an authenticated session, then hand off branch
`codex/mmlongbench-ocr2-313` and run ID `20260717T182300Z` to the local
website-construction agent.

## Result Return

Package only validated text/JSON OCR artifacts under
`sol_results/mmlongbench_ocr2_unstructured_313/<RUN_ID>/`, commit them, and push
this branch. Record job IDs, the run path, and the pushed commit SHA here.

## Active Jobs And Blockers

- Render job `59167963`: completed (`0:0`), 313 unique rendered pages validated.
- Smoke job `59168210`: completed (`0:0`), page
  `mmlongbench_page_94c4dea17c77f92613a4` passed with 7 segments.
- Attempt-1 array `59168867`: all 32 tasks completed (`0:0`), 313 unique runs.
- Quality-gate job `59177847`: completed (`0:0`), 306 valid and 7 excluded pages.
- Incomplete-package job `59178879`: completed (`0:0`) in 1:08 on `sc001`.
- Result commit: `a59d47f` (`data: add MMLongBench OCR2 exploratory results`).
- Blocker: SOL cannot push over the configured HTTPS remote because no GitHub
  credentials are available and the graphical askpass prompt cannot open.
