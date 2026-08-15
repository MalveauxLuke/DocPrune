# MMLongBench-Doc OCR2 Segmentation Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare a Git-backed, 313-page MMLongBench-Doc pilot that a SOL agent can render and run through pinned DeepSeek-OCR-2, then push the validated OCR outputs back without constructing semantic sections or a webpage on SOL.

**Architecture:** A tracked pilot-data directory freezes ten PDFs, their source rows, and a 313-row page plan. SOL renders those pages into scratch using the already installed PyMuPDF dependency, runs the existing sharded DeepSeek-OCR-2 adapter, reuses the existing conservative attempt validation, and packages only portable text/JSON OCR artifacts under `sol_results/`. A short `sol/CURRENT_SOL_TASK.md` becomes the sole active handoff; the previous segment-evidence handoff/spec set moves to the archive.

**Tech Stack:** Python 3.12, PyMuPDF, PyArrow for one-time parquet extraction, pytest, existing DeepSeek-OCR-2 runner, Slurm, Git.

## Global Constraints

- Process exactly 10 PDFs and exactly 313 pages.
- Reject any selected PDF with more than 50 pages; never truncate a PDF.
- Use only the ten filenames and SHA-256 values frozen in `sol/task_spec/mmlongbench_deepseek_ocr2_segmentation_pilot.md`.
- Use `deepseek-ai/DeepSeek-OCR-2` revision `aaa02f3811945a91062062994c5c4a3f4c0af2b0` with the existing pinned inference settings.
- Reuse `scripts/sciegqa_parser_compare/deepseek_runner.py`; do not add a second inference implementation.
- SOL may render and run OCR, but must not construct semantic sections, labels, viewer manifests, HTML, or a web server.
- Do not commit rendered PNGs, caches, model files, environments, or Slurm output.
- Return OCR artifacts through a scoped commit on `codex/mmlongbench-ocr2-313`; never push to `main`.
- Execute this plan in an isolated worktree because the current checkout contains unrelated user changes.
- Preserve all unrelated working-tree content and stage only task-owned paths.

---

## File Structure

**Create:**

- `scripts/mmlongbench_ocr2_pilot.py` — frozen inventory, manifest generation, deterministic rendering, attempt selection, and portable result packaging.
- `scripts/prepare_mmlongbench_ocr2_pilot.py` — CLI over the pilot module.
- `tests/test_mmlongbench_ocr2_pilot.py` — manifest, rendering, packaging, and failure-gate tests.
- `tests/test_mmlongbench_ocr2_sol.py` — static SOL-wrapper and handoff contract tests.
- `sol/render_mmlongbench_ocr2_313.sbatch` — CPU rendering job.
- `sol/run_mmlongbench_ocr2_313.sbatch` — sharded GPU inference job.
- `sol/finalize_mmlongbench_ocr2_313.sbatch` — retry-manifest, merge, audit, and package job.
- `pilot_data/mmlongbench_ocr2_unstructured_313/` — tracked PDFs and frozen input manifests.
- `sol/archive/colqwen/segment_evidence_20260716/` — superseded active handoff/specification material.

**Modify:**

- `sol/CURRENT_SOL_TASK.md` — replace with the short MMLongBench pilot handoff.
- `.gitignore` — ignore only scratch-style local render/viewer payloads if the implementation introduces them; do not ignore `pilot_data/` or `sol_results/`.

**Reuse unchanged:**

- `scripts/run_sciegqa_deepseek_ocr.py`
- `scripts/sciegqa_parser_compare/deepseek_runner.py`
- `scripts/sciegqa_evidence_poc.py` public attempt-validation helpers
- `environments/deepseek-ocr2-sol.yml`
- `sol/setup_deepseek_ocr.sbatch`

---

### Task 1: Freeze the ten-document input contract

**Files:**

- Create: `scripts/mmlongbench_ocr2_pilot.py`
- Create: `scripts/prepare_mmlongbench_ocr2_pilot.py`
- Create: `tests/test_mmlongbench_ocr2_pilot.py`
- Create: `pilot_data/mmlongbench_ocr2_unstructured_313/documents/*`
- Create: `pilot_data/mmlongbench_ocr2_unstructured_313/documents.jsonl`
- Create: `pilot_data/mmlongbench_ocr2_unstructured_313/questions.jsonl`
- Create: `pilot_data/mmlongbench_ocr2_unstructured_313/page_plan.jsonl`
- Create: `pilot_data/mmlongbench_ocr2_unstructured_313/source_snapshot.json`

**Interfaces:**

- Produces: `prepare_pilot_data(source_documents_dir: Path, source_parquet: Path, output_root: Path) -> dict[str, object]`
- Produces: a deterministic `page_plan.jsonl` consumed by Task 2.
- Depends on: `artifact_sha256`, `make_stable_id`, and `write_jsonl` from `scripts.sciegqa_mineru.schema`.

- [ ] **Step 1: Write failing inventory and manifest tests**

Add tests with these assertions:

```python
from scripts.mmlongbench_ocr2_pilot import (
    DOCUMENT_SPECS,
    build_page_plan,
    select_questions,
    validate_document_inventory,
)


def test_frozen_inventory_is_ten_documents_and_313_pages() -> None:
    assert len(DOCUMENT_SPECS) == 10
    assert sum(item.page_count for item in DOCUMENT_SPECS) == 313
    assert max(item.page_count for item in DOCUMENT_SPECS) <= 50
    assert len({item.filename for item in DOCUMENT_SPECS}) == 10
    assert all(len(item.sha256) == 64 for item in DOCUMENT_SPECS)


def test_page_plan_is_complete_stable_and_one_based(tmp_path: Path) -> None:
    documents = [
        {"filename": item.filename, "sha256": item.sha256, "page_count": item.page_count}
        for item in DOCUMENT_SPECS
    ]
    forward = build_page_plan(documents)
    reverse = build_page_plan(list(reversed(documents)))
    assert forward == reverse
    assert len(forward) == 313
    assert len({row["page_id"] for row in forward}) == 313
    assert min(row["page_number"] for row in forward) == 1
    assert all("image_path" not in row for row in forward)
    assert all("image_sha256" not in row for row in forward)


def test_question_selection_keeps_all_rows_for_selected_documents() -> None:
    rows = [
        {"doc_id": DOCUMENT_SPECS[0].filename, "question": "kept"},
        {"doc_id": "research-paper.pdf", "question": "excluded"},
    ]
    assert [row["question"] for row in select_questions(rows)] == ["kept"]
```

Add negative tests proving `validate_document_inventory` rejects a hash mismatch,
page-count mismatch, an 11th document, a document above 50 pages, and a total
other than 313.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
python -m pytest -q tests/test_mmlongbench_ocr2_pilot.py --tb=short
```

Expected: collection fails with `ModuleNotFoundError: scripts.mmlongbench_ocr2_pilot`.

- [ ] **Step 3: Implement the frozen inventory and pure manifest functions**

Use this public shape:

```python
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.sciegqa_mineru.schema import artifact_sha256, make_stable_id, write_jsonl


DATASET_REVISION = "2ff6aa9237fc777b6627dc57a486e9225ac5fb86"
PARQUET_SHA256 = "bcdac3c96669634c34184814cede4fe57cf7ac0f98dde0e85936394f6a56a02d"
EXPECTED_DOCUMENT_COUNT = 10
EXPECTED_PAGE_COUNT = 313
MAX_DOCUMENT_PAGES = 50


@dataclass(frozen=True)
class DocumentSpec:
    filename: str
    page_count: int
    sha256: str
    document_type: str


DOCUMENT_SPECS: tuple[DocumentSpec, ...] = (
    DocumentSpec("91521110100M_4K_UHD_Display_User_Manual_V1.1.pdf", 40, "a3dfbac95e5c99ea9aa9fb830bd5f338d3278c7df28a775bc314acedc22b20d5", "display_user_manual"),
    DocumentSpec("owners-manual-2170416.pdf", 32, "91e70aaf49192b4deb640154a50d0b14976c831be11a7f858f9e953a761b8681", "refrigerator_owners_manual"),
    DocumentSpec("mi_phone.pdf", 30, "16ee1cfbd58f9c6a3793c66ff5f4ec38dd864ac9fd7a9b7186f8af7db53f870b", "mobile_phone_guide"),
    DocumentSpec("watch_d.pdf", 27, "bb5fd3576ac080c867f200ddc90311cdb351fb8240c70c9468dc433936c136e6", "smartwatch_guide"),
    DocumentSpec("ISEP_student_handbook_2020.pdf", 24, "27163e80a71d12f267592d87c1112413ac85ed432c766f45e9558a47e6f9cb80", "student_handbook"),
    DocumentSpec("camry_ebrochure.pdf", 26, "d3f8049bc0c48b735b7c4e0d6399ed9e8188afd332f58829f44e1c39cfb13528", "automotive_brochure"),
    DocumentSpec("honor_watch_gs_pro.pdf", 42, "2d5559e514dc54f616ffea8d8e054117e32da672bdbf9797883dd815fa990ee5", "smartwatch_guide"),
    DocumentSpec("StudentSupport_Guidebook.pdf", 44, "7515cd264f0a33a782169d938a291e9ec4fc81a094a30a4ecf951d4cfa7ff336", "student_support_guidebook"),
    DocumentSpec("GPL-Graduate-Studies-Professional-Learning-Brochure-Jul-2021.pdf", 17, "83424ce04cc38c567e121a1f6827e370a878917a8d8b5b7743e05f1c49042f5b", "graduate_studies_brochure"),
    DocumentSpec("NUS-FASS-Graduate-Guidebook-2021-small.pdf", 31, "8e7802bfbb70e4126bcc6a5f4d06bcc1c549dd24cf4c90a143eab8971abe66a8", "graduate_guidebook"),
)


def build_page_plan(documents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    order = {item.filename: index for index, item in enumerate(DOCUMENT_SPECS)}
    rows: list[dict[str, Any]] = []
    for document in sorted(documents, key=lambda row: order[str(row["filename"])]):
        for page_number in range(1, int(document["page_count"]) + 1):
            rows.append({
                "page_id": make_stable_id("mmlongbench_page", document["sha256"], page_number),
                "document_filename": document["filename"],
                "document_sha256": document["sha256"],
                "document_page_count": document["page_count"],
                "page_number": page_number,
            })
    return rows


def select_questions(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = {item.filename for item in DOCUMENT_SPECS}
    return sorted(
        (dict(row) for row in rows if str(row.get("doc_id")) in selected),
        key=lambda row: (str(row["doc_id"]), str(row["question"])),
    )
```

Implement PDF validation with `fitz.open(path).page_count`, verify exact bytes and
hashes before copying, reject an existing nonempty output directory, and write
all JSON/JSONL deterministically with sorted keys. Load PyArrow only inside the
CLI preparation path so importing the module does not require PyArrow.

- [ ] **Step 4: Add the preparation CLI**

`scripts/prepare_mmlongbench_ocr2_pilot.py` must expose:

```bash
python scripts/prepare_mmlongbench_ocr2_pilot.py prepare-data \
  --source-documents-dir "/Users/god/Documents/document vqa/MMLongBench-Doc/documents" \
  --source-parquet "/Users/god/Documents/document vqa/MMLongBench-Doc/data/train-00000-of-00001.parquet" \
  --output-root pilot_data/mmlongbench_ocr2_unstructured_313
```

The CLI must print a JSON audit and exit nonzero on any failed invariant. The
audit must report exactly 10 documents, 313 pages, and 68 source questions.

- [ ] **Step 5: Run tests, prepare the real tracked dataset, and verify it**

Run:

```bash
python -m pytest -q tests/test_mmlongbench_ocr2_pilot.py --tb=short
python scripts/prepare_mmlongbench_ocr2_pilot.py prepare-data \
  --source-documents-dir "/Users/god/Documents/document vqa/MMLongBench-Doc/documents" \
  --source-parquet "/Users/god/Documents/document vqa/MMLongBench-Doc/data/train-00000-of-00001.parquet" \
  --output-root pilot_data/mmlongbench_ocr2_unstructured_313
wc -l pilot_data/mmlongbench_ocr2_unstructured_313/{documents,questions,page_plan}.jsonl
```

Expected line counts: `10`, `68`, and `313`.

- [ ] **Step 6: Commit the input contract**

```bash
git add scripts/mmlongbench_ocr2_pilot.py \
  scripts/prepare_mmlongbench_ocr2_pilot.py \
  tests/test_mmlongbench_ocr2_pilot.py \
  pilot_data/mmlongbench_ocr2_unstructured_313
git commit -m "feat: freeze MMLongBench OCR2 pilot inputs"
```

---

### Task 2: Render all 313 pages deterministically on SOL

**Files:**

- Modify: `scripts/mmlongbench_ocr2_pilot.py`
- Modify: `scripts/prepare_mmlongbench_ocr2_pilot.py`
- Modify: `tests/test_mmlongbench_ocr2_pilot.py`
- Create: `sol/render_mmlongbench_ocr2_313.sbatch`
- Create: `tests/test_mmlongbench_ocr2_sol.py`

**Interfaces:**

- Consumes: tracked `page_plan.jsonl` and `documents/` from Task 1.
- Produces: scratch `<RUN_ROOT>/selection/pages.jsonl` with 313 image paths,
  hashes, dimensions, and PyMuPDF provenance.

- [ ] **Step 1: Write failing rendering tests**

Create a two-page synthetic PDF with PyMuPDF and assert:

```python
def test_render_page_plan_writes_hashed_png_manifest(tmp_path: Path) -> None:
    # Build a two-page PDF under documents/, construct matching plan rows, then:
    result = render_page_plan(page_plan, documents_root, tmp_path / "run", matrix_scale=2.0)
    assert len(result) == 2
    assert [row["page_number"] for row in result] == [1, 2]
    assert all(Path(row["image_path"]).is_file() for row in result)
    assert all(len(row["image_sha256"]) == 64 for row in result)
    assert all(row["render_matrix_scale"] == 2.0 for row in result)
    assert all(row["renderer"] == "PyMuPDF" for row in result)
```

Also test rejection of duplicate page IDs, a missing page-plan row, a source hash
mismatch, an out-of-range page number, and a nonempty render destination.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
python -m pytest -q tests/test_mmlongbench_ocr2_pilot.py::test_render_page_plan_writes_hashed_png_manifest --tb=short
```

Expected: failure because `render_page_plan` does not exist.

- [ ] **Step 3: Implement deterministic rendering**

Implement this behavior in `scripts/mmlongbench_ocr2_pilot.py`:

```python
def render_page_plan(
    page_plan: Sequence[Mapping[str, Any]],
    documents_root: Path,
    run_root: Path,
    *,
    matrix_scale: float = 2.0,
) -> list[dict[str, Any]]:
    import fitz

    pages_dir = run_root / "pages"
    manifest_path = run_root / "selection/pages.jsonl"
    if pages_dir.exists() or manifest_path.exists():
        raise ValueError("render destination already exists")
    pages_dir.mkdir(parents=True)
    matrix = fitz.Matrix(matrix_scale, matrix_scale)
    rendered: list[dict[str, Any]] = []
    for row in page_plan:
        pdf_path = documents_root / str(row["document_filename"])
        if artifact_sha256(pdf_path) != row["document_sha256"]:
            raise ValueError(f"source PDF hash mismatch: {pdf_path.name}")
        with fitz.open(pdf_path) as document:
            page_number = int(row["page_number"])
            if document.page_count != int(row["document_page_count"]):
                raise ValueError(f"source PDF page-count mismatch: {pdf_path.name}")
            pixmap = document.load_page(page_number - 1).get_pixmap(matrix=matrix, alpha=False)
            image_path = pages_dir / f'{row["page_id"]}.png'
            pixmap.save(image_path)
        rendered.append({
            **dict(row),
            "image_path": str(image_path.resolve()),
            "image_sha256": artifact_sha256(image_path),
            "image_bytes": image_path.stat().st_size,
            "image_width": pixmap.width,
            "image_height": pixmap.height,
            "renderer": "PyMuPDF",
            "renderer_version": fitz.VersionBind,
            "render_matrix_scale": matrix_scale,
        })
    write_jsonl(manifest_path, rendered)
    return rendered
```

Validate the full 313-row page-ID set before writing the manifest. Do not retain
open PDF handles between documents.

- [ ] **Step 4: Add the render CLI and SOL wrapper**

CLI:

```bash
python scripts/prepare_mmlongbench_ocr2_pilot.py render \
  --pilot-root "$PROJECT_DIR/pilot_data/mmlongbench_ocr2_unstructured_313" \
  --run-root "$RUN_ROOT" \
  --matrix-scale 2.0
```

`sol/render_mmlongbench_ocr2_313.sbatch` must use `lightwork`, four CPUs, 16 GiB,
two hours, `/home/$USER/mamba-envs/deepseek-ocr2-sol/bin/python`,
`PYTHONNOUSERSITE=1`, and scratch logs. It must print the project commit and
tracked manifest hashes before rendering.

- [ ] **Step 5: Add wrapper contract tests and run them**

Assert the render wrapper contains all required resources and paths, contains no
GPU request, and invokes only the `render` CLI. Run:

```bash
python -m pytest -q tests/test_mmlongbench_ocr2_pilot.py tests/test_mmlongbench_ocr2_sol.py --tb=short
```

Expected: all tests pass.

- [ ] **Step 6: Commit deterministic rendering**

```bash
git add scripts/mmlongbench_ocr2_pilot.py \
  scripts/prepare_mmlongbench_ocr2_pilot.py \
  sol/render_mmlongbench_ocr2_313.sbatch \
  tests/test_mmlongbench_ocr2_pilot.py tests/test_mmlongbench_ocr2_sol.py
git commit -m "feat: render MMLongBench pilot pages on SOL"
```

---

### Task 3: Add the SOL OCR, retry, and portable packaging workflow

**Files:**

- Modify: `scripts/mmlongbench_ocr2_pilot.py`
- Modify: `scripts/prepare_mmlongbench_ocr2_pilot.py`
- Modify: `tests/test_mmlongbench_ocr2_pilot.py`
- Modify: `tests/test_mmlongbench_ocr2_sol.py`
- Create: `sol/run_mmlongbench_ocr2_313.sbatch`
- Create: `sol/finalize_mmlongbench_ocr2_313.sbatch`

**Interfaces:**

- Consumes: `<RUN_ROOT>/selection/pages.jsonl` from Task 2.
- Reuses: `build_retry_manifest`, `merge_attempts`, `load_attempt_shards`, and
  `validate_ocr_attempt` from `scripts.sciegqa_evidence_poc`.
- Produces: `sol_results/mmlongbench_ocr2_unstructured_313/<RUN_ID>/` with
  portable relative paths and a complete hash inventory.

- [ ] **Step 1: Write failing result-packaging tests**

Build a temporary merged-run fixture with two valid pages and assert:

```python
def test_package_ocr_results_rewrites_paths_and_hashes_every_file(tmp_path: Path) -> None:
    output = package_ocr_results(
        run_root=merged_run,
        page_plan_path=page_plan_path,
        output_root=tmp_path / "sol_results" / "run_1",
        expected_page_count=2,
    )
    runs = read_jsonl(output / "deepseek_ocr2/runs.jsonl")
    assert len(runs) == 2
    assert all(not Path(row["raw_output_path"]).is_absolute() for row in runs)
    assert (output / "completion_audit.json").is_file()
    assert (output / "artifact_manifest.json").is_file()
    assert (output / "SHA256SUMS").is_file()
    assert verify_sha256sums(output) == []
```

Add negative tests for 312/314 pages, page-plan/render/run ID disagreement,
terminal OCR failures, duplicate run or segment IDs, hash mismatch, absolute
paths left in the packaged manifests, an existing output directory, and any PNG
or PDF entering the package.

- [ ] **Step 2: Run the focused packaging test and verify it fails**

Run:

```bash
python -m pytest -q tests/test_mmlongbench_ocr2_pilot.py::test_package_ocr_results_rewrites_paths_and_hashes_every_file --tb=short
```

Expected: failure because `package_ocr_results` does not exist.

- [ ] **Step 3: Implement portable result packaging**

`package_ocr_results` must:

1. Read the tracked `page_plan.jsonl`, returned `selection/pages.jsonl`, merged
   `deepseek_ocr2/runs.jsonl`, merged `segments.jsonl`, and parse audit.
2. Require exact equality of all page-ID sets and the expected cardinality.
3. Require every run to have `status="completed"`, `selected_attempt` in `{1,2}`,
   and valid recorded artifact hashes.
4. Copy only `grounded_output.txt`, `document.md`, and
   `parse_diagnostics.json`; regenerate each portable `run.json` from the
   rewritten run row.
5. Rewrite run artifact paths and segment provenance paths to POSIX paths
   relative to the package root.
6. Copy the environment freeze when present.
7. Write `pages.jsonl` and `completion_audit.json`, then write
   `artifact_manifest.json` for every result payload except the manifest itself
   and `SHA256SUMS`.
8. Write lexicographically ordered `SHA256SUMS` for every package file except
   `SHA256SUMS` itself, including `artifact_manifest.json`, then verify it
   immediately.
9. Reject file suffixes `.png`, `.pdf`, model/cache directories, and Slurm logs.
10. Build in a sibling staging directory and use `os.replace` only after every
    gate passes.

The completion audit must include:

```json
{
  "document_count": 10,
  "page_count": 313,
  "completed_page_count": 313,
  "terminal_failure_count": 0,
  "first_attempt_count": 0,
  "retry_count": 0,
  "page_plan_sha256": "...",
  "rendered_pages_sha256": "...",
  "runs_sha256": "...",
  "segments_sha256": "..."
}
```

Populate attempt counts from the selected runs rather than hardcoding them.

- [ ] **Step 4: Add the GPU inference wrapper**

`sol/run_mmlongbench_ocr2_313.sbatch` must be a pilot-specific adaptation of
`sol/run_sciegqa_4k_deepseek_ocr2.sbatch` with:

```text
#SBATCH --array=0-31%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
```

It must require `RUN_ROOT`, `PAGES_MANIFEST`, and `ATTEMPT_INDEX`; use the
existing DeepSeek environment and CLI; write attempt shards beneath
`$RUN_ROOT/deepseek_ocr2_attempts/attempt_$ATTEMPT_INDEX/shard_NNN`; and remove
all SciEGQA selection-query assumptions.

- [ ] **Step 5: Add the retry/finalize wrapper**

`sol/finalize_mmlongbench_ocr2_313.sbatch` must use `lightwork` and require
`MODE`. Supported modes:

```text
MODE=build-retry   -> call existing build_retry_manifest and write retry_pages.jsonl
MODE=merge-package -> call existing merge_attempts, require zero terminal pages,
                      then package into the tracked sol_results path
```

The retry GPU submission must pass
`PAGES_MANIFEST=$RUN_ROOT/selection/retry_pages.jsonl` and `ATTEMPT_INDEX=2`.
The finalizer must not run Git commands; the SOL agent performs the explicit
stage/commit/push gate after reviewing the package.

- [ ] **Step 6: Run focused tests**

Run:

```bash
python -m pytest -q \
  tests/test_mmlongbench_ocr2_pilot.py \
  tests/test_mmlongbench_ocr2_sol.py \
  tests/test_deepseek_grounding.py \
  tests/test_sciegqa_parser_sol.py \
  --tb=short
```

Expected: all tests pass without downloading model weights.

- [ ] **Step 7: Commit the SOL execution and Git-return workflow**

```bash
git add scripts/mmlongbench_ocr2_pilot.py \
  scripts/prepare_mmlongbench_ocr2_pilot.py \
  sol/run_mmlongbench_ocr2_313.sbatch \
  sol/finalize_mmlongbench_ocr2_313.sbatch \
  tests/test_mmlongbench_ocr2_pilot.py tests/test_mmlongbench_ocr2_sol.py
git commit -m "feat: add MMLongBench OCR2 SOL workflow"
```

---

### Task 4: Archive the old SOL task and install the sole active handoff

**Files:**

- Move: `sol/CURRENT_SOL_TASK.md` content to `sol/archive/colqwen/segment_evidence_20260716/CURRENT_SOL_TASK.md`
- Move: `sol/sol_stage0_stage1_agent_handoff.md`
- Move: `sol/task_spec/current_design_architecture_plan.md`
- Move: `sol/task_spec/segment_evidence_experiment_plan.md`
- Move: all tracked `sol/*.bak-*` and `sol/task_spec/*.bak-*`
- Create: `sol/CURRENT_SOL_TASK.md`
- Modify: `tests/test_mmlongbench_ocr2_sol.py`

**Interfaces:**

- Produces: the only active SOL handoff, pointing to the approved pilot spec.
- Preserves: old task history under `sol/archive/colqwen/segment_evidence_20260716/`.

- [ ] **Step 1: Write failing active-task tests**

Add:

```python
def test_mmlongbench_pilot_is_the_sole_active_sol_task() -> None:
    current = Path("sol/CURRENT_SOL_TASK.md").read_text()
    assert "mmlongbench_deepseek_ocr2_segmentation_pilot.md" in current
    assert "313" in current
    assert "SOL must not build" in current
    assert not list(Path("sol").glob("*.bak-*"))
    assert not list(Path("sol/task_spec").glob("*.bak-*"))
    assert not Path("sol/task_spec/current_design_architecture_plan.md").exists()
    assert not Path("sol/task_spec/segment_evidence_experiment_plan.md").exists()
```

- [ ] **Step 2: Run the test and verify it fails against the old handoff**

Run:

```bash
python -m pytest -q tests/test_mmlongbench_ocr2_sol.py::test_mmlongbench_pilot_is_the_sole_active_sol_task --tb=short
```

Expected: failure because the old segment-evidence task is still active.

- [ ] **Step 3: Archive superseded files with Git history preserved**

Use `git mv`, not copy/delete pairs. Place every superseded file beneath:

```text
sol/archive/colqwen/segment_evidence_20260716/
```

Keep its original basename unless two backups collide; for collisions, prefix
the original parent directory (`sol_` or `task_spec_`). Do not move Slurm
wrappers, environments, `sol/sol_logs.md`, or experiment source code.

- [ ] **Step 4: Replace `sol/CURRENT_SOL_TASK.md` with a short handoff**

Use this structure:

```markdown
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
- Fixed scope: 10 PDFs, 313 pages, every PDF at most 50 pages.
- SOL renders and runs OCR only. SOL must not build semantic sections or a webpage.

## Next Action

Validate the tracked inputs, submit the render smoke/full job, run one OCR smoke
page, then submit attempt 1 according to the binding specification.

## Result Return

Package only validated text/JSON OCR artifacts under
`sol_results/mmlongbench_ocr2_unstructured_313/<RUN_ID>/`, commit them, and push
this branch. Record job IDs, the run path, and the pushed commit SHA here.

## Active Jobs And Blockers

- Jobs: none submitted.
- Blockers: none known before SOL preflight.
```

- [ ] **Step 5: Run the active-task and archive tests**

Run:

```bash
python -m pytest -q tests/test_mmlongbench_ocr2_sol.py --tb=short
find sol sol/task_spec -maxdepth 1 -type f -name '*.bak-*' -print
```

Expected: tests pass and `find` prints nothing.

- [ ] **Step 6: Commit the handoff replacement**

```bash
git add sol tests/test_mmlongbench_ocr2_sol.py
git commit -m "docs: hand off MMLongBench OCR2 pilot to SOL"
```

---

### Task 5: Verify and publish the prepared pilot branch

**Files:**

- Verify all task-owned files from Tasks 1-4.
- Do not modify or stage unrelated paths.

**Interfaces:**

- Produces: remote branch `codex/mmlongbench-ocr2-313` ready for the SOL agent.
- Completion proof: repository tests, manifest counts/hashes, Git inventory, and
  remote branch SHA.

- [ ] **Step 1: Run the focused verification suite**

Run:

```bash
python -m pytest -q \
  tests/test_mmlongbench_ocr2_pilot.py \
  tests/test_mmlongbench_ocr2_sol.py \
  tests/test_deepseek_grounding.py \
  tests/test_sciegqa_parser_sol.py \
  --tb=short
```

Expected: all tests pass.

- [ ] **Step 2: Revalidate the real input manifests and assets**

Run:

```bash
python scripts/prepare_mmlongbench_ocr2_pilot.py validate-data \
  --pilot-root pilot_data/mmlongbench_ocr2_unstructured_313
wc -l pilot_data/mmlongbench_ocr2_unstructured_313/{documents,questions,page_plan}.jsonl
find pilot_data/mmlongbench_ocr2_unstructured_313/documents -type f -name '*.pdf' | wc -l
```

Expected: JSON audit reports 10 documents, 68 questions, and 313 pages; line
counts are `10`, `68`, `313`; PDF count is `10`.

- [ ] **Step 3: Audit branch contents and repository size impact**

Run:

```bash
git status --short
git diff origin/COLQWEN_binary_classification...HEAD --name-status
find pilot_data/mmlongbench_ocr2_unstructured_313/documents -type f -size +95M -print
git count-objects -vH
```

Expected: no PDF exceeds 95 MiB; only planned files and the already approved
design commits differ from the remote base; unrelated user files are absent
because execution occurred in the isolated worktree.

- [ ] **Step 4: Push the dedicated branch**

Run:

```bash
git branch --show-current
git push -u origin codex/mmlongbench-ocr2-313
```

Expected: current branch is exactly `codex/mmlongbench-ocr2-313` and the push
succeeds. If authentication fails, stop after recording the local branch and
HEAD SHA; do not change remotes or push to another branch.

- [ ] **Step 5: Provide the SOL launch prompt**

Give the user only:

```text
Pull branch codex/mmlongbench-ocr2-313, then read AGENTS.md and
sol/CURRENT_SOL_TASK.md and execute the current task exactly as written.
```

Do not restate the detailed procedure in the launch prompt; the handoff is the
source of truth.
