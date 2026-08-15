# SciEGQA 4K SOL Run

The historical requirements are in
`sol/archive/sciegqa/SCIEGQA_4K_HANDOFF.md`. This archived run
uses `Yuwh07/SciEGQA-Train` at revision
`4ffb867c88e3264161920b4b2446d5ac6352269e`, seed `20260630`, and the approved
domain quotas. `query_intent` is an auditable lexical query-text proxy; it is
never reported as observed evidence modality.

Large inputs and outputs live under `/scratch/$USER/sciegqa_train_4k`. The
tracked repository contains only code, tests, submit scripts, the self-authored
fixture, and documentation.

## Submit scripts

- `sol/archive/jobs/sciegqa/download_sciegqa_train_images.sbatch`: pinned,
  resumable archive download.
- `sol/archive/jobs/sciegqa/extract_sciegqa_4k_pages.sbatch`: one-pass archive-member validation and
  selective extraction of chosen PNGs.
- `sol/archive/jobs/sciegqa/setup_mineru34.sbatch`: isolated Python 3.12 / MinerU 3.4 / Torch / vLLM
  environment setup with GPU preflight.
- `sol/archive/jobs/sciegqa/run_mineru34_smoke.sbatch`: localhost-only API, bounded health polling,
  artifact validation, canonical geometry checks, and EXIT cleanup.

## Synthetic fixture

`archive/tests/fixtures/mineru_smoke/mineru_smoke.pdf` is generated deterministically by
`generate_fixture.py`; its SHA-256 is recorded in the adjacent `SHA256SUMS`.
It contains original paragraph, table, line-figure/caption, and equation content.

The smoke submit script defaults to this PDF. A real selected page uses the same
script with `INPUT_KIND=image`, `INPUT_PATH=<validated PNG>`,
`RUN_LABEL=real_page`, and a page-appropriate `REQUIRED_TYPES` value. Full 4K
MinerU processing is intentionally out of scope.
