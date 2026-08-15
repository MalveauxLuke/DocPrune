# SciEGQA MinerU vs DeepSeek-OCR Initial Comparison Implementation Plan and SOL Handoff

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan in the existing worktree. Do not dispatch subagents. Track the checkbox steps in this document, batch the work, and make one scoped commit only after the final verification gate.

**Goal:** Build and run a reproducible 32-page SciEGQA comparison of MinerU 3.4, DeepSeek-OCR, and DeepSeek-OCR-2 using identical validated PNG inputs, then expose gold evidence and all parser boxes in a visually verifiable web viewer.

**Architecture:** Select 32 anchor queries on 32 unique pages from the validated 4,000-query manifests, with exactly four anchor pages per domain and deterministic query-intent coverage. Run each parser independently and preserve its raw output. Convert parser outputs into one provenance-preserving normalized schema, compute geometry/structure/operational metrics separately, and build a new comparison viewer without changing the completed MinerU pilot viewer.

**Tech Stack:** Python 3.12, pytest, MinerU 3.4.0 hybrid engine, PyTorch 2.6, Transformers 4.46.3, FlashAttention 2.7.3, DeepSeek-OCR, DeepSeek-OCR-2, Slurm, static HTML/CSS/JavaScript.

---

## 1. Decision record

The approved initial experiment is deliberately bounded to 32 unique pages. It is not permission to parse all 3,711 pages.

Fixed decisions:

- Source dataset: `Yuwh07/SciEGQA-Train`.
- Dataset revision: `4ffb867c88e3264161920b4b2446d5ac6352269e`.
- Existing 4K selection: 4,000 strict single-page, single-region queries; 3,711 unique pages.
- Input unit: the already validated page PNG, not a separately rendered PDF.
- Sample size: 32 unique pages.
- Domain allocation: exactly four anchor pages from each of `q-fin`, `q-bio`, `eess`, `physics`, `cs`, `econ`, `stat`, and `math`.
- Parser set:
  - MinerU `3.4.0`, current repository configuration.
  - `deepseek-ai/DeepSeek-OCR` at revision `9f30c71f441d010e5429c532364a86705536c53a`.
  - `deepseek-ai/DeepSeek-OCR-2` at revision `aaa02f3811945a91062062994c5c4a3f4c0af2b0`.
- DeepSeek prompt: `<image>\n<|grounding|>Convert the document to markdown.`
- DeepSeek-OCR settings: `base_size=1024`, `image_size=640`, `crop_mode=True`, `eval_mode=True`, `save_results=False`.
- DeepSeek-OCR-2 settings: `base_size=1024`, `image_size=768`, `crop_mode=True`, `eval_mode=True`, `save_results=False`.
- Correctness path: Transformers inference first. vLLM throughput work is explicitly deferred until the grounded output is correct.
- Environment isolation: OCR 1 and OCR 2 get separate absolute Mamba environments. Neither may modify `/home/$USER/mamba-envs/mineru34-sol`.
- Generated datasets, model caches, logs, and parser outputs stay under `/scratch/$USER` and are not committed.
- Existing files under `viewer/sciegqa_mineru/` are not modified. The comparison gets a separate viewer.
- Figure deep parsing with `Parse the figure.` is not part of this first test because it is a second model call and changes the comparison unit.

## 2. Why this is not a one-score benchmark

MinerU emits detected and post-processed blocks. DeepSeek emits a generated sequence containing Markdown and optional grounding tokens. The comparison must therefore keep three metric families separate:

1. **Geometry:** bounding-box validity, gold coverage, IoU, segment precision, fragmentation, and union coverage.
2. **Structure:** reading order, type labels, caption association, Markdown preservation, tables, formulas, and generation defects.
3. **Operations:** elapsed time, peak GPU memory, output tokens, failures, retries, truncation, and repetition.

Do not publish a single aggregate “winner” score. Report the three families independently and retain the page-level records used to calculate them.

## 3. Required reading on SOL

Before changing code or submitting jobs, read:

1. `AGENTS.md`
2. `SOLinstrucitons.md`
3. `sol/CURRENT_SOL_TASK.md`
4. `sol/SCIEGQA_4K_RUN.md`
5. `sol/SCIEGQA_4K_HANDOFF.md`
6. Every file in `sol/bug_fixes/`
7. `docs/mineru_local_setup.md`
8. `scripts/sciegqa_4k/pipeline.py`
9. `scripts/document_parsing/schema.py`
10. `scripts/sciegqa_mineru/mineru_runner.py`
11. `scripts/sciegqa_mineru/canonicalize.py`
12. `scripts/sciegqa_mineru/viewer_bundle.py`
13. `viewer/sciegqa_mineru/app.js`
14. `sol/setup_mineru34.sbatch`
15. `sol/run_mineru34_smoke.sbatch`

Then replace `sol/CURRENT_SOL_TASK.md` with only the current comparison state, next action, current job IDs, scratch paths, and blockers. Delete the completed 4K setup history from that current-state file; it remains documented elsewhere.

Primary external references:

- DeepSeek-OCR paper: <https://arxiv.org/pdf/2510.18234>
- DeepSeek-OCR-2 paper: <https://arxiv.org/pdf/2601.20552>
- OCR 1 repository: <https://github.com/deepseek-ai/DeepSeek-OCR>
- OCR 2 repository: <https://github.com/deepseek-ai/DeepSeek-OCR-2>
- vLLM OCR recipe: <https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html>

## 4. Repository and scratch layout

### Tracked files to create

```text
environments/
  deepseek-ocr1-sol.yml
  deepseek-ocr2-sol.yml
scripts/
  build_sciegqa_parser_pilot.py
  run_sciegqa_parser_mineru.py
  run_sciegqa_deepseek_ocr.py
  evaluate_sciegqa_parser_pilot.py
  build_sciegqa_parser_viewer.py
  sciegqa_parser_compare/
    __init__.py
    selection.py
    grounding.py
    deepseek_runner.py
    metrics.py
    viewer_bundle.py
sol/
  setup_deepseek_ocr.sbatch
  run_sciegqa_parser_mineru.sbatch
  run_sciegqa_deepseek_ocr.sbatch
viewer/
  sciegqa_parser_compare/
    index.html
    app.js
    styles.css
tests/
  test_sciegqa_parser_selection.py
  test_deepseek_grounding.py
  test_sciegqa_parser_metrics.py
  test_sciegqa_parser_viewer.py
  test_sciegqa_parser_sol.py
```

Modify only when necessary:

- `.gitignore`: ignore a local parser-comparison output directory only if the implementation creates one inside the checkout.
- `sol/CURRENT_SOL_TASK.md`: SOL current state only.

Do not modify the completed 4K manifests, current MinerU viewer, frozen verifier code, or unrelated tests.

### Generated scratch layout

```text
/scratch/$USER/sciegqa_train_4k/parser_compare/<RUN_ID>/
  selection/
    anchors.jsonl
    pages.jsonl
    queries.jsonl
    selection_audit.json
  mineru/
    runs.jsonl
    segments.jsonl
    raw/<page_id>/...
  deepseek_ocr1/
    runs.jsonl
    segments.jsonl
    raw/<page_id>/grounded_output.txt
    raw/<page_id>/document.md
    raw/<page_id>/parse_diagnostics.json
  deepseek_ocr2/
    runs.jsonl
    segments.jsonl
    raw/<page_id>/grounded_output.txt
    raw/<page_id>/document.md
    raw/<page_id>/parse_diagnostics.json
  evaluation/
    page_metrics.jsonl
    parser_summary.json
    parser_summary.md
    visual_review.jsonl
  site/
    index.html
    app.js
    styles.css
    manifest.json
    pages/<page_id>.png
```

Use a UTC identifier such as `20260630T210000Z`. Never overwrite a prior run.

## 5. Canonical contracts

### Anchor and page selection

`anchors.jsonl` contains exactly 32 rows and uses one selected query to explain why each unique page entered the pilot:

```json
{
  "anchor_rank": 1,
  "query_id": "query_...",
  "page_id": "page_...",
  "category": "cs",
  "query_intent": "plot_chart",
  "reasoning_operation": "numeric_lookup",
  "source_record_index": 123,
  "source_bbox_norm_1000": [100.0, 200.0, 800.0, 700.0],
  "selection_seed": 20260630
}
```

`queries.jsonl` contains every 4K query associated with the 32 chosen pages, not just the anchors. Add `is_anchor`; this yields extra gold-box evaluations without extra parser inference. Report anchor-only metrics and all-linked-query metrics separately.

The selector must:

- validate exactly 4,000 input query rows and 3,711 input page rows;
- validate the pinned dataset revision on every query;
- validate every selected PNG exists and its relative path matches `pages/<category>/<doc>/<doc>_<page>.png` under the existing scratch root;
- choose exactly four unique pages per domain;
- never choose two anchors from the same page;
- represent every query-intent class that exists in the 4K selection at least once globally;
- use the 4K query-intent distribution as the target after the rare-class floor;
- break all ties by `source_record_index`, then `query_id`;
- write an audit containing domain counts, intent counts, linked-query count, unique-page count, and SHA-256 values for all source manifests.

Use deterministic constrained round-robin selection, not random search:

1. Group queries by domain and intent.
2. Sort each group by `(source_record_index, query_id)`.
3. Allocate four slots per domain by largest remainder over that domain's intent distribution.
4. Apply a global rare-intent floor by transferring a slot from the most overrepresented intent while preserving four slots per domain.
5. Walk the selected domain/intent queues, skipping any already-selected page.
6. Fail if the constraints cannot be satisfied; do not weaken them silently.

### Parser segment

Every parser writes the same outer schema while preserving parser-specific fields:

```json
{
  "segment_id": "segment_...",
  "page_id": "page_...",
  "run_id": "run_...",
  "parser": "deepseek_ocr2",
  "parser_version": "aaa02f3811945a91062062994c5c4a3f4c0af2b0",
  "reading_order": 0,
  "type": "figure",
  "raw_type": "image",
  "bbox_norm_1000": [10.0, 20.0, 900.0, 800.0],
  "content": {"text": "...", "markdown": "..."},
  "provenance": {
    "raw_output_path": ".../grounded_output.txt",
    "raw_start": 0,
    "raw_end": 83,
    "annotation_index": 0,
    "box_index": 0
  }
}
```

Rules:

- `parser` is one of `mineru`, `deepseek_ocr1`, `deepseek_ocr2`.
- Keep `raw_type` unchanged. `type` may use an explicit versioned mapping, with unknown labels mapped to `unknown`, never guessed.
- Validate all boxes using `BBox.validate((1000, 1000))`.
- A grounding annotation containing multiple boxes becomes multiple segments sharing `annotation_index` and content, with distinct `box_index` values.
- Preserve character offsets into the immutable raw output.
- Stable IDs include parser, model revision, page ID, annotation index, box index, and box coordinates.
- Never replace DeepSeek coordinates with MinerU coordinates or vice versa.

### DeepSeek grounding grammar

Parse the generated form:

```text
<|ref|>LABEL<|/ref|><|det|>[[x0,y0,x1,y1]]<|/det|>CONTENT
```

`CONTENT` runs until the next valid `<|ref|>` annotation or end of output. Implement a state-machine scanner rather than one greedy regular expression. It must record, without crashing:

- unmatched `<|ref|>` or `<|det|>` tags;
- invalid JSON coordinate payloads;
- nonnumeric coordinates;
- reversed or out-of-range boxes;
- annotations with no box;
- empty labels;
- ungrounded leading or trailing content.

The conservative `document.md` view strips only the grounding wrappers and label tokens while retaining generated content. It is a derived artifact; `grounded_output.txt` remains authoritative.

### Run provenance

Each parser run row records:

- page ID and input PNG SHA-256;
- parser and exact model revision;
- exact prompt and inference parameters;
- repository git revision;
- Python executable and `pip freeze` SHA-256;
- Slurm job ID and GPU identity;
- start/completion UTC timestamps and elapsed seconds;
- peak allocated GPU memory;
- raw output path and SHA-256;
- generated token count where available;
- status and explicit failure reason;
- whether an existing completed artifact was reused.

Resume only when the stored input hash, model revision, prompt, parameters, and artifact hashes all match. Otherwise fail and require a new run directory.

## 6. Implementation tasks

### Task 1: Deterministic 32-page selector

**Files:**

- Create `scripts/sciegqa_parser_compare/selection.py`
- Create `scripts/build_sciegqa_parser_pilot.py`
- Create `tests/test_sciegqa_parser_selection.py`

- [ ] Write tests proving exact cardinality, four anchors per domain, 32 unique pages, global intent coverage, determinism under reversed input order, linked-query inclusion, and fail-closed source-count/revision checks.
- [ ] Run `python -m pytest tests/test_sciegqa_parser_selection.py -q` and confirm the tests fail because the module is absent.
- [ ] Implement `select_anchor_pages(queries, pages, seed=20260630)` and `write_pilot_manifests(...)` according to Section 5.
- [ ] Add the CLI:

```bash
python scripts/build_sciegqa_parser_pilot.py \
  --selection-dir /scratch/$USER/sciegqa_train_4k/selection \
  --pages-root /scratch/$USER/sciegqa_train_4k/pages \
  --output-dir "$RUN_ROOT/selection"
```

- [ ] Rerun the focused tests and expect all to pass.
- [ ] Run the CLI twice into separate temporary directories and require identical SHA-256 values for all four output files.

### Task 2: Grounded-output parser

**Files:**

- Create `scripts/document_parsing/grounding.py`
- Create `tests/test_deepseek_grounding.py`

Required public interface:

```python
@dataclass(frozen=True)
class GroundingDiagnostic:
    code: str
    message: str
    raw_start: int
    raw_end: int

def parse_grounded_output(
    raw: str,
    *,
    page_id: str,
    run_id: str,
    parser: str,
    parser_version: str,
    raw_output_path: str,
) -> tuple[list[dict[str, object]], list[GroundingDiagnostic]]: ...

def grounding_to_markdown(raw: str) -> str: ...
```

- [ ] Test one annotation, multiple annotations, multiple boxes, content containing brackets, malformed JSON, invalid boxes, missing closing tags, empty output, and stable raw offsets.
- [ ] Verify the tests fail before implementation.
- [ ] Implement the scanner and conservative Markdown conversion.
- [ ] Verify the focused tests pass.

### Task 3: DeepSeek inference adapter

**Files:**

- Create `scripts/document_parsing/deepseek_runner.py`
- Create `scripts/document_parsing/run_deepseek_ocr.py`
- Add adapter tests to `tests/test_deepseek_grounding.py`

The adapter must load exactly one approved model per process:

```python
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    revision=model_revision,
    trust_remote_code=True,
)
model = AutoModel.from_pretrained(
    model_name,
    revision=model_revision,
    trust_remote_code=True,
    use_safetensors=True,
    _attn_implementation="flash_attention_2",
).eval().cuda().to(torch.bfloat16)
```

Call the pinned remote-code method with `eval_mode=True`. This is essential because the method returns the unmodified decoded generation in that mode. Do not use `save_results=True` as the primary data path because its post-processing removes grounding annotations.

- [ ] Test configuration validation and resume validation using a fake model/tokenizer; unit tests must not download weights.
- [ ] Implement atomic per-page writes for `grounded_output.txt`, `document.md`, `parse_diagnostics.json`, and the run record.
- [ ] Record `torch.cuda.max_memory_allocated()` after resetting peak statistics for each page.
- [ ] Detect likely truncation when generation reaches 8,192 new tokens or ends with an incomplete grounding tag.
- [ ] Detect repetition using repeated normalized 30-token windows and record it as a diagnostic rather than editing the output.
- [ ] Verify a failure row is written before the process exits nonzero; do not continue past CUDA OOM or model-load failure.

### Task 4: MinerU 32-page adapter

**Files:**

- Create `scripts/run_sciegqa_parser_mineru.py`
- Add focused tests to `tests/test_sciegqa_parser_metrics.py`

Reuse, rather than duplicate:

- `build_mineru_image_command()` and `run_page()` from `scripts/sciegqa_mineru/mineru_runner.py`;
- `canonicalize_content_list()` and `canonicalize_middle_captions()` from `scripts/sciegqa_mineru/canonicalize.py`;
- `BBox`, `artifact_sha256`, `make_stable_id`, `read_jsonl`, and `write_jsonl` from `scripts/document_parsing/schema.py`.

- [ ] Test conversion of fixture content-list and middle-caption data into `parser="mineru"` records.
- [ ] Implement a loop over exactly the 32 selected page images while one local MinerU API remains alive.
- [ ] Preserve the full native MinerU artifact inventory and SHA-256 values.
- [ ] Fail if a page produces no valid canonical segments.
- [ ] Use the already approved MinerU configuration: `hybrid-engine`, `high`, image analysis, formula, and table enabled.

### Task 5: Geometry, structure, and operational metrics

**Files:**

- Create `scripts/sciegqa_parser_compare/metrics.py`
- Create `scripts/evaluate_sciegqa_parser_pilot.py`
- Create `tests/test_sciegqa_parser_metrics.py`

For each query/parser pair calculate:

```text
best_iou
best_gold_coverage = intersection / gold_area
best_segment_precision = intersection / segment_area
intersecting_segment_count
union_gold_coverage
smallest_covering_segment_area_ratio
has_valid_segment
```

For each page/parser record:

```text
segment_count
type_counts
invalid_grounding_count
ungrounded_content_chars
markdown_chars
table_marker_count
formula_marker_count
likely_truncated
repetition_detected
elapsed_seconds
peak_gpu_memory_bytes
status
```

Aggregate all linked queries and the 32 anchors separately. The summary must include distributions, not only means: count, minimum, p50, p95, maximum, and mean.

- [ ] Test containment, partial overlap, fragmentation, disjoint boxes, union coverage, and zero-segment pages with hand-calculated fixtures.
- [ ] Verify tests fail, implement the metrics, and verify they pass.
- [ ] Write both `parser_summary.json` and a concise `parser_summary.md` that explicitly refuses to collapse the metric families into a single score.

### Task 6: Comparison viewer

**Files:**

- Create `scripts/sciegqa_parser_compare/viewer_bundle.py`
- Create `scripts/build_sciegqa_parser_viewer.py`
- Create `viewer/sciegqa_parser_compare/index.html`
- Create `viewer/sciegqa_parser_compare/app.js`
- Create `viewer/sciegqa_parser_compare/styles.css`
- Create `tests/test_sciegqa_parser_viewer.py`

Required modes:

```text
Gold only
MinerU only
DeepSeek-OCR only
DeepSeek-OCR-2 only
Gold + MinerU
Gold + OCR 1
Gold + OCR 2
All parsers
```

Viewer requirements:

- Use the original page PNG with an SVG overlay in `viewBox="0 0 1000 1000"`.
- Give each parser a distinct color and line style; gold remains solid red.
- Provide independent type filters per parser.
- Show raw type, normalized type, box, reading order, content, parser revision, and overlap metrics on hover/focus.
- Show the selected query, answer, domain, source row, page ID, anchor status, and all linked queries.
- Allow query navigation and page navigation without duplicating page assets.
- Expose the conservative DeepSeek Markdown in a side panel while retaining a link/name for the raw grounded artifact.
- Never use a filled rectangle that obscures scientific content; use transparent outlines.
- Escape all generated text with DOM `textContent`; do not inject Markdown with `innerHTML`.
- Reject duplicate IDs, missing parser results, invalid boxes, missing page images, or unknown page references while building the manifest.

Tests must cover manifest joins, all eight modes, parser/type filtering, invalid-box rejection, and one page with multiple linked queries.

### Task 7: Pinned SOL environments

**Files:**

- Create `environments/deepseek-ocr1-sol.yml`
- Create `environments/deepseek-ocr2-sol.yml`
- Create `sol/setup_deepseek_ocr.sbatch`
- Create `tests/test_sciegqa_parser_sol.py`

Both YAML files pin:

```text
python=3.12.9
pip
transformers==4.46.3
tokenizers==0.20.3
PyMuPDF
img2pdf
einops
easydict
addict
Pillow
numpy
huggingface-hub
```

The setup job accepts only `OCR_VERSION=1` or `OCR_VERSION=2` and maps them to:

```text
1 -> /home/$USER/mamba-envs/deepseek-ocr1-sol
2 -> /home/$USER/mamba-envs/deepseek-ocr2-sol
```

After creating the base environment, install the official CUDA stack inside that environment:

```bash
"$PYTHON_BIN" -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118
"$PYTHON_BIN" -m pip install flash-attn==2.7.3 --no-build-isolation
```

The job must set `PYTHONNOUSERSITE=1` before environment resolution, use an absolute interpreter, export the environment library paths, print `nvidia-smi`, verify BF16 support, import all pinned packages, and write `pip freeze` to scratch.

Do not add vLLM to either first-test environment. The official Transformers path is the correctness baseline, and avoiding vLLM removes a known version-conflict variable.

### Task 8: Slurm inference jobs

**Files:**

- Create `sol/run_sciegqa_parser_mineru.sbatch`
- Create `sol/run_sciegqa_deepseek_ocr.sbatch`
- Extend `tests/test_sciegqa_parser_sol.py`

MinerU job requirements:

- one GPU, four CPUs, 48 GB RAM, four-hour limit on `public/public`;
- absolute MinerU environment and Python paths;
- localhost API with a job-derived port;
- bounded health polling and `trap cleanup EXIT`;
- exactly the selected 32 pages;
- no full-4K mode.

DeepSeek job requirements:

- one GPU, four CPUs, 64 GB RAM, four-hour limit on `public/public`;
- `OCR_VERSION` validation before environment activation;
- exact model and revision mapping inside the script;
- model/data caches on scratch;
- one model loaded per job and pages processed sequentially;
- resume only verified completed pages;
- no network-served API and no user-site packages.

Static tests must reject missing pins, mutable `main`, missing `PYTHONNOUSERSITE`, missing cleanup, relative Python resolution, or any command that targets all 3,711 pages.

### Task 9: Local verification before SOL

- [ ] Run:

```bash
python -m pytest \
  tests/test_sciegqa_parser_selection.py \
  tests/test_deepseek_grounding.py \
  tests/test_sciegqa_parser_metrics.py \
  tests/test_sciegqa_parser_viewer.py \
  tests/test_sciegqa_parser_sol.py -q
```

- [ ] Expected result: all new focused tests pass.
- [ ] Run the existing related tests:

```bash
python -m pytest \
  tests/test_sciegqa_4k_selection.py \
  tests/test_sciegqa_4k_pipeline.py \
  tests/test_sciegqa_4k_sol.py \
  tests/test_sciegqa_mineru_canonicalize.py \
  tests/test_sciegqa_mineru_viewer_bundle.py \
  tests/test_sciegqa_mineru_viewer_static.py -q
```

- [ ] Confirm `git diff --check` is clean.
- [ ] Confirm generated weights, pages, parser outputs, and logs are absent from `git status --short`.

## 7. SOL execution sequence

Run only after the tracked implementation passes locally.

### 7.1 Preflight and sample materialization

```bash
cd "$HOME/COLPALI_binary_classification"
git branch --show-current
git rev-parse HEAD
git status --short
df -h "/scratch/$USER"

export ROOT="/scratch/$USER/sciegqa_train_4k"
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ROOT="$ROOT/parser_compare/$RUN_ID"
mkdir -p "$RUN_ROOT" "$ROOT/logs"

python scripts/build_sciegqa_parser_pilot.py \
  --selection-dir "$ROOT/selection" \
  --pages-root "$ROOT/pages" \
  --output-dir "$RUN_ROOT/selection"
```

Expected audit invariants:

```text
anchors=32
unique_pages=32
domain_counts={each domain: 4}
all source files and images present
```

### 7.2 Environment jobs

```bash
OCR1_SETUP_JOB=$(OCR_VERSION=1 ROOT="$ROOT" sbatch --parsable sol/setup_deepseek_ocr.sbatch)
OCR2_SETUP_JOB=$(OCR_VERSION=2 ROOT="$ROOT" sbatch --parsable sol/setup_deepseek_ocr.sbatch)
echo "$OCR1_SETUP_JOB $OCR2_SETUP_JOB"
```

Do not start inference until both setup jobs complete successfully and their import/BF16 preflights pass.

### 7.3 One-page correctness smokes

Use the first anchor page and separate `smoke/` output directories. Run OCR 1, OCR 2, and MinerU. Verify before batching:

- raw output is nonempty;
- the exact pinned model revision is recorded;
- at least one valid box is parsed for DeepSeek;
- raw `<|ref|>` and `<|det|>` tokens are preserved;
- `document.md` exists but does not replace the raw output;
- MinerU content-list, middle JSON, and canonical segments exist;
- every box passes normalized-coordinate validation.

If either DeepSeek model produces Markdown but no grounding tokens, stop. Do not silently switch to `Free OCR.` because that removes the geometry required by this experiment.

### 7.4 Batch jobs

```bash
MINERU_JOB=$(RUN_ROOT="$RUN_ROOT" ROOT="$ROOT" \
  sbatch --parsable sol/run_sciegqa_parser_mineru.sbatch)

OCR1_JOB=$(OCR_VERSION=1 RUN_ROOT="$RUN_ROOT" ROOT="$ROOT" \
  sbatch --parsable sol/run_sciegqa_deepseek_ocr.sbatch)

OCR2_JOB=$(OCR_VERSION=2 RUN_ROOT="$RUN_ROOT" ROOT="$ROOT" \
  sbatch --parsable sol/run_sciegqa_deepseek_ocr.sbatch)

echo "MinerU=$MINERU_JOB OCR1=$OCR1_JOB OCR2=$OCR2_JOB"
```

Monitor with `squeue`, `sacct`, and the exact scratch logs. Update `sol/CURRENT_SOL_TASK.md` with current job IDs and blockers, not a running diary.

### 7.5 Evaluation and site build

After all three jobs succeed:

```bash
python scripts/evaluate_sciegqa_parser_pilot.py --run-root "$RUN_ROOT"
python scripts/build_sciegqa_parser_viewer.py \
  --run-root "$RUN_ROOT" \
  --viewer-source viewer/sciegqa_parser_compare \
  --site-dir "$RUN_ROOT/site"
```

Fail unless each parser has exactly 32 completed page runs and no selected page is missing from the viewer manifest.

## 8. Visual verification gate

Serve only on localhost:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory "$RUN_ROOT/site"
```

Tunnel from the workstation:

```bash
ssh -N -L 8765:127.0.0.1:8765 "$USER@sol.asu.edu"
```

Open <http://127.0.0.1:8765/>.

Every one of the 32 pages must be inspected in all of these states:

1. Gold only.
2. MinerU only.
3. DeepSeek-OCR only.
4. DeepSeek-OCR-2 only.
5. All parsers overlaid.

Record one `visual_review.jsonl` row per page:

```json
{
  "page_id": "page_...",
  "reviewer": "...",
  "reviewed_at": "2026-06-30T00:00:00Z",
  "gold_faithful": true,
  "mineru_faithful": true,
  "deepseek_ocr1_faithful": true,
  "deepseek_ocr2_faithful": true,
  "all_overlay_alignment_checked": true,
  "notes": ""
}
```

“Faithful” means the displayed rectangle matches the stored normalized coordinates on the original PNG. It does not mean the parser made the semantically correct segmentation; semantic mistakes belong in `notes` and the metrics.

The final validator must require exactly 32 unique review rows and all five boolean fields. A false value is a real failed gate requiring investigation; do not redefine it as success.

## 9. Completion criteria

The initial test is complete only when:

- the deterministic sample contains 32 unique pages and exactly four anchors per domain;
- all source revisions, hashes, and image paths validate;
- MinerU, OCR 1, and OCR 2 each have 32 verified completed runs;
- all raw parser outputs and canonical segments are preserved;
- metrics exist for anchors and all linked queries;
- no parser is reduced to one aggregate score;
- the viewer exposes all eight modes and all parser/type filters;
- every page passes the visual alignment review gate;
- focused and related regression tests pass;
- `git diff --check` passes;
- generated datasets, model files, logs, and run outputs remain on scratch;
- one final scoped commit contains only implementation, tests, submit scripts, environment definitions, and this handoff.

Do not claim completion from successful Slurm exit codes alone.

## 10. Genuine blockers

Stop and report evidence if any of these occurs:

- the live 4K manifests do not reproduce 4,000 queries and 3,711 pages;
- a selected PNG hash/path cannot be validated;
- the allocated GPU lacks BF16 or cannot load either pinned model;
- FlashAttention 2.7.3 cannot build against the allocated SOL CUDA/toolchain;
- grounded mode consistently produces no grounding tokens on the one-page smoke;
- an exact model revision is unavailable;
- completing the comparison would require parsing the full 3,711-page set;
- faithful browser verification cannot be performed through a localhost tunnel.

Do not mask these with a different model revision, unpinned package upgrade, `Free OCR.`, CPU fallback, or altered sample size without user approval.
