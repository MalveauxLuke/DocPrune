# SciEGQA 4K DeepSeek-OCR2 Evidence Dataset: SOL Handoff

## Objective

Build a conservative proof-of-concept dataset for:

```text
known gold page + query -> fine-grained evidence section
```

Use the existing stratified 4,000-query SciEGQA-Train subset and its 3,711
unique gold pages. Give each unique page one primary inference attempt with
pinned DeepSeek-OCR2, retry only failed or invalid attempts once, build
semantic sections with the exact native logic used by the
checked-in `Gold + final labeling` viewer, assign query-relative positive and
same-page hard-negative relations at gold coverage `0.70`, quarantine every
ambiguous page, and create deterministic page-grouped 70/15/15 splits.

This is a page-conditional evidence-localization dataset. It is not an
open-corpus page-retrieval benchmark.

## Non-negotiable decisions

1. The source is the existing 4,000-query selection, not all 30,780 source
   rows.
2. OCR is processed per unique gold page, not per query: one primary attempt
   and at most one retry after failure or invalid output. The selected valid
   result is reused by every query on that page.
3. DeepSeek-OCR2 is the only parser used for final semantic sections.
4. Semantic sections must use the same shared implementation as the viewer.
   Do not copy, approximate, or reimplement the rules in a second code path.
5. Gold coverage threshold is exactly `0.70`.
6. Every retained query must have exactly one positive section.
7. Every other section on that query's page is a hard negative for that query.
8. Labels are query-section relations. Never store `positive` or
   `hard_negative` on a shared section record.
9. A section positive for one query may be a hard negative for another query
   on the same page.
10. If any query-section relation on a page is partial, quarantine the entire
    page and all its queries.
11. If any query has zero or multiple positives, quarantine the entire page
    and all its queries.
12. Retry a failed or invalid OCR inference once. If the second attempt is not
    valid, quarantine the page and all its queries.
13. Split only after quarantine: 70% train, 15% validation, 15% test, grouped
    by page and approximately stratified by domain and query intent.
14. Pages may not cross splits. Documents may cross splits.
15. Preserve original page paths and exact member boxes. Do not render,
    union-crop, or stitch standalone section images in this phase.
16. Do not impose paragraph, token, area, or section-length limits.
17. Keep all datasets, model caches, raw OCR, logs, and generated manifests on
    `/scratch/$USER`. Commit only code, tests, submit scripts, and this handoff.

## Required repository state

The verified final-label implementation currently lives on
`codex/sciegqa-final-labeling`. Before SOL work begins:

1. Merge that branch into `COLQWEN_binary_classification`.
2. Push `COLQWEN_binary_classification`.
3. Pull it on SOL.
4. Record the exact merged commit with `git rev-parse HEAD`.

The SOL commit must contain commit `546d9c8` in its ancestry. Verify:

```bash
git merge-base --is-ancestor 546d9c8 HEAD
```

If that command fails, stop. The checkout does not contain the verified
`figure_title` compatibility for both images and tables.

## Required reading on SOL

Read these before editing or submitting jobs:

1. `AGENTS.md`
2. `SOLinstrucitons.md`
3. `sol/CURRENT_SOL_TASK.md`
4. `sol/SCIEGQA_4K_HANDOFF.md`
5. `sol/SCIEGQA_4K_RUN.md`
6. `sol/README.md`
7. `sol/SOL_RUN_GUIDE.md`
8. Every file in `sol/bug_fixes/`
9. `scripts/document_parsing/semantic_sections.py`
10. `scripts/document_parsing/grounding.py`
11. `scripts/document_parsing/deepseek_runner.py`
12. `scripts/document_parsing/run_deepseek_ocr.py`
13. `sol/run_sciegqa_deepseek_ocr.sbatch`
14. `scripts/build_sciegqa_4k.py`
15. `scripts/extract_sciegqa_4k_pages.py`
16. `tests/test_sciegqa_final_labeling.py`
17. `tests/test_deepseek_grounding.py`

Replace `sol/CURRENT_SOL_TASK.md` with the live state for this task. Keep only
the current state, next action, job IDs, paths, and blockers; do not append a
history log.

## Immutable inputs

Dataset:

- Repository: `Yuwh07/SciEGQA-Train`
- Revision: `4ffb867c88e3264161920b4b2446d5ac6352269e`
- Selection seed: `20260630`
- Selected queries: exactly `4,000`
- Selected unique pages: exactly `3,711`

DeepSeek-OCR2:

- Model: `deepseek-ai/DeepSeek-OCR-2`
- Revision: `aaa02f3811945a91062062994c5c4a3f4c0af2b0`
- Prompt: `<image>\n<|grounding|>Convert the document to markdown.`
- Base size: `1024`
- Image size: `768`
- Crop mode: enabled
- Evaluation mode: enabled
- Maximum generated tokens: `8192`

Expected existing inputs:

```text
/scratch/$USER/sciegqa_train_4k/
  raw/SciEGQA-Train.jsonl
  raw/images.tar
  selection/queries.jsonl
  selection/pages.jsonl
  selection/documents.jsonl
  selection/selection_audit.json
  selection/extraction_audit.json
  pages/<category>/<doc_name>/<doc_name>_<page>.png
```

`selection/pages.jsonl` must already contain `image_path`, image dimensions,
image byte size, and image SHA-256 from the completed extraction workflow.

## Preflight invariants

Run lightweight inspections on the login node; run Python validation inside a
compute allocation.

```bash
export PROJECT_DIR="$HOME/COLPALI_binary_classification"
export ROOT="/scratch/$USER/sciegqa_train_4k"
cd "$PROJECT_DIR"

git branch --show-current
git rev-parse HEAD
git merge-base --is-ancestor 546d9c8 HEAD
wc -l "$ROOT/selection/queries.jsonl"
wc -l "$ROOT/selection/pages.jsonl"
df -h "/scratch/$USER"
```

Required results:

- branch: `COLQWEN_binary_classification`;
- 4,000 query rows;
- 3,711 page rows;
- every query joins to exactly one page;
- every page image exists and matches its recorded SHA-256;
- all gold boxes validate in `[0,1000]` coordinates;
- dataset and model revisions match the pinned values;
- enough scratch space exists for raw outputs and model cache.

Stop on any mismatch. Do not regenerate the 4K selection or silently adjust
counts.

## Output layout

Create one immutable run directory:

```bash
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ROOT="$ROOT/evidence_poc/$RUN_ID"
mkdir -p "$RUN_ROOT"/{selection,deepseek_ocr2_attempts,deepseek_ocr2,labels,splits,quarantine,audits,logs}
```

Final layout:

```text
$RUN_ROOT/
  provenance.json
  selection/
    queries.jsonl
    pages.jsonl
  deepseek_ocr2_attempts/
    shard_000/
    ...
  deepseek_ocr2/
    runs.jsonl
    segments.jsonl
    raw/<page_id>/...
    environment_freeze.txt
  labels/
    semantic_sections.jsonl
    queries.jsonl
    query_section_labels.jsonl
  splits/
    page_assignments.jsonl
    train.jsonl
    validation.jsonl
    test.jsonl
  quarantine/
    pages.jsonl
    queries.jsonl
  audits/
    parse_audit.json
    labeling_audit.json
    split_audit.json
    determinism.json
  logs/
```

Copy the selection manifests into `$RUN_ROOT/selection` and record their
SHA-256 values. Never mutate the source selection in `$ROOT/selection`.

## Phase 1: generalize the proven OCR runner

The current parser runner contains a hard-coded 32-page invariant and writes a
single shared `runs.jsonl`/`segments.jsonl`. Do not run that implementation
concurrently against 3,711 pages.

Use test-driven development to make these targeted changes:

1. Allow an explicit pages manifest instead of assuming the 32-page pilot.
2. Accept deterministic `--shard-index` and `--shard-count` arguments.
3. Stable-sort pages by `page_id`, then assign a page to a shard when its
   stable sorted position modulo `shard_count` equals `shard_index`.
4. Write each shard to its own directory. No two jobs may write the same JSONL
   or raw-output directory.
5. Preserve the existing per-page raw output, markdown, diagnostics, run
   record, artifact hashes, model revision, prompt, inference parameters, GPU
   identity, Slurm job ID, environment freeze, and resume validation.
6. Remove only the 32-page cardinality assumption. Do not change model or
   grounding behavior.
7. Keep completed valid pages resumable by exact image hash and inference
   provenance.
8. Preserve every inference attempt. A retry must not overwrite the first
   attempt's raw output or diagnostics.

Add focused tests for:

- deterministic, exhaustive, disjoint sharding;
- non-32-page manifests;
- mismatched resume provenance;
- isolated shard output paths;
- no duplicate page or segment IDs;
- preservation of failed first-attempt artifacts.

## Phase 2: sharded DeepSeek-OCR2 inference

The verified 32-page run averaged approximately 26.9 seconds per page, so
3,711 pages represent roughly 28 aggregate GPU-hours before retry overhead.
Use 32 deterministic shards with a conservative array concurrency cap of 8.
Each shard contains approximately 116 pages.

Create a dedicated array submit script based on
`sol/run_sciegqa_deepseek_ocr.sbatch`:

```text
sol/run_sciegqa_4k_deepseek_ocr2.sbatch
```

Required Slurm/runtime properties:

- `#SBATCH --array=0-31%8`
- one GPU per task;
- public partition/QoS;
- four CPU cores;
- 64 GB RAM;
- four-hour wall time;
- absolute scratch logs;
- `set -euo pipefail`;
- `PYTHONNOUSERSITE=1`;
- absolute environment path `/home/$USER/mamba-envs/deepseek-ocr2-sol`;
- scratch `HF_HOME` and Transformers cache;
- explicit `LD_LIBRARY_PATH` from the environment;
- print Python, Torch, Transformers, GPU, BF16 support, model, revision, git
  commit, shard index, and input-manifest hashes before inference.

Submit from the repository checkout:

```bash
export PROJECT_DIR="$HOME/COLPALI_binary_classification"
export ROOT="/scratch/$USER/sciegqa_train_4k"
export RUN_ROOT="$ROOT/evidence_poc/$RUN_ID"

OCR_ARRAY_JOB_ID=$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",ROOT="$ROOT",RUN_ROOT="$RUN_ROOT" \
  sol/run_sciegqa_4k_deepseek_ocr2.sbatch)
echo "$OCR_ARRAY_JOB_ID"
```

Record the job ID immediately in `sol/CURRENT_SOL_TASK.md`.

## Phase 3: retry and OCR quality gate

After the first array completes, validate each of the 3,711 page attempts.

A page attempt is valid only when all are true:

- run status is `completed`;
- model revision, prompt, parameters, and input PNG hash match;
- raw output and markdown files exist and match recorded hashes;
- generated tokens are below the configured maximum;
- `likely_truncated` is false;
- `repetition_detected` is false;
- parse diagnostics are empty;
- grounding tag counts are balanced;
- at least one canonical segment exists;
- every segment box is finite and valid in `[0,1000]`;
- segment IDs and reading orders are internally valid.

Build a retry pages manifest containing every page that failed this gate.
Submit the same sharded runner once more into separate attempt directories.
Do not submit a third attempt.

After the retry:

- select the first valid attempt for each page;
- if the first attempt was invalid and the second is valid, retain both and
  mark the second as selected;
- if neither attempt is valid, mark the page `ocr_quality_failed` for
  quarantine;
- never generate semantic sections from an invalid attempt.

## Phase 4: deterministic merge

Create a CPU merge/validation command and an `afterok` finalization job. It
must:

1. Read only isolated shard outputs.
2. Validate exactly one selected valid attempt or one terminal quarantine
   record for each of 3,711 page IDs.
3. Reject duplicate page IDs, run IDs, segment IDs, or raw artifact paths.
4. Stable-sort merged runs by `page_id`.
5. Stable-sort segments by `(page_id, reading_order, segment_id)`.
6. Write merged files atomically.
7. Verify every selected raw artifact and manifest hash after writing.
8. Record accepted first attempts, accepted retries, terminal failures,
   elapsed-time distribution, token distribution, and diagnostic counts in
   `audits/parse_audit.json`.

Do not allow multiple array tasks to append to the merged files.

## Phase 5: exact semantic-section generation

The viewer and dataset builder must call one shared semantic-section function
from `scripts/document_parsing/semantic_sections.py`. If necessary, expose
the existing private candidate-set helper as a public function and make both
callers use it. Do not duplicate its contents.

For this dataset, call only the verified `deepseek_native` strategy with the
same `350.0` floating-title geometry limit used by the current viewer. Do not
generate the geometry-only comparison candidate set and do not generate the
five viewer thresholds.

The exact native section rules are:

### Headed text pages

- Markdown headings are `##` through `######`.
- A heading starts one semantic section extending to the next heading or page
  end.
- Include eligible text and formula members belonging to that section.
- Never split a headed section by length, token count, paragraph count,
  columns, area, or bounding-box geometry.
- Preserve an eligible document preamble before the first `##` heading as its
  own semantic section.

### Pages without headings

- Use atomic DeepSeek title, text, and formula paragraph units.
- Exclude native visual titles from paragraph candidates.
- Do not merge or split paragraphs by length.

### Images, tables, and titles

- Native visual raw types are `image` and `table`.
- DeepSeek raw type `figure_title` is compatible with both images and tables.
- Discover visual runs top-down.
- Consecutive same-kind visual blocks separated by at most two raw `text`
  blocks form one run; ignored text is not a member of the visual bundle.
- For each run, search immediately upward first for one free compatible
  `figure_title`.
- If an upward title is found, claim it and stop the primary search.
- If no upward title is found, search immediately downward.
- A primary title can be claimed by only one run.
- Unclaimed `figure_title` blocks use the 350-unit center-distance geometry
  fallback across images and tables and may merge connected visual runs.
- Preserve every member segment ID, member box, reading order, raw type, and
  link-method field.

### Geometry

- Candidate geometry is the union of member boxes, not the bounding envelope.
- Gold coverage is intersection with the gold box divided by gold-box area.
- Candidate precision may be recorded for analysis but does not determine the
  label.

Before the 4K run, add a golden regression that rebuilds the checked-in
32-page viewer's native candidates and proves that candidate membership,
member boxes, caption links, text modes, and query labels remain unchanged.

## Phase 6: query-relative labeling and page quarantine

Use threshold `0.70` exactly:

```text
gold_coverage >= 0.70  -> positive
0 < gold_coverage < 0.70 -> partial
gold_coverage == 0 -> hard_negative
```

Apply labels independently for every `(query_id, section_id)` pair on the
query's gold page.

Quarantine the entire page and every query on it when any condition holds:

- terminal OCR-quality failure;
- any partial query-section relation;
- any query has zero positives;
- any query has more than one positive.

For a retained page:

- each query has exactly one positive relation;
- every remaining section is a hard negative for that query;
- no cross-page negatives are created;
- all same-page hard negatives are retained; do not sample them in dataset
  generation;
- a section may have different labels for different queries.

Write page quarantine reasons as sorted, explicit codes. Preserve affected
query IDs, selected/failed run IDs, candidate IDs, coverage values, and source
provenance. Never send partial cases to manual review during this POC; they are
excluded but fully auditable.

## Phase 7: relational schemas

### `labels/semantic_sections.jsonl`

One unlabeled row per shared semantic section:

- `section_id`
- `page_id`
- `kind`: `headed_text`, `document_preamble`, `deepseek_paragraph`, or
  `visual_bundle`
- `source_parser`
- `source_parser_version`
- `heading`
- `text`
- `markdown`
- `member_segment_ids`
- `member_bboxes_norm_1000`
- `member_reading_orders`
- `member_types`
- `member_raw_types`
- `caption_link`
- `word_count`
- page image path and image SHA-256
- selected OCR run ID and raw-artifact hashes
- repository and dataset revisions

Do not include a class label on this row.

### `labels/queries.jsonl`

One row per retained query, preserving the existing 4K query provenance plus:

- `page_id`
- `positive_section_id`
- `split`
- retained section count
- hard-negative count

Preserve the answer for provenance and auditing, but mark it explicitly as not
a model input.

### `labels/query_section_labels.jsonl`

One row per query-section relation:

- stable `pair_id`
- `query_id`
- `page_id`
- `section_id`
- `split`
- `label`: `positive` or `hard_negative`
- `gold_coverage`
- `candidate_precision`
- `intersection_area`
- `candidate_union_area`
- gold box
- section member boxes
- dataset revision, model revision, and repository commit

No `partial` row may appear in this file.

### Quarantine manifests

`quarantine/pages.jsonl` contains one row per excluded page with sorted reason
codes, all query IDs, and complete parse/label diagnostics.

`quarantine/queries.jsonl` contains one row per excluded query with its page
quarantine reasons and original 4K provenance.

## Phase 8: page-grouped 70/15/15 split

Split only retained pages. Use seed `20260630`.

Requirements:

1. Group all queries and query-section relations by `page_id`.
2. Assign each page to exactly one of `train`, `validation`, or `test`.
3. Target query proportions of 70%, 15%, and 15% after quarantine.
4. Approximately preserve domain and query-intent distributions by split.
5. Use stable sorting and a seeded deterministic grouped-stratification
   algorithm; never depend on set or filesystem iteration order.
6. Same-document pages may be assigned to different splits.
7. Report achieved page, query, positive-pair, and negative-pair counts, plus
   per-domain and per-intent deviations from targets.
8. Fail if any page, query, or pair appears in multiple splits.

Write:

- `splits/page_assignments.jsonl`: one page-to-split relation;
- `splits/train.jsonl`: retained query-section relations for training pages;
- `splits/validation.jsonl`: retained relations for validation pages;
- `splits/test.jsonl`: retained relations for test pages.

Split files should reference shared sections rather than duplicating OCR text,
markdown, or bounding-box arrays.

## Phase 9: tests and verification

Add focused tests before implementation for:

- exact reuse of the viewer semantic-section builder;
- headed sections remain unsplit regardless of length;
- no-heading paragraph fallback;
- document preamble preservation;
- image runs and table runs using `figure_title`;
- upward-first primary title ownership;
- two-text-block run bridging and three-text-block stopping;
- floating-title geometry merge;
- member-union gold coverage;
- threshold boundaries at `0`, just above `0`, just below `0.70`, and `0.70`;
- page quarantine on any partial relation;
- page quarantine on zero or multiple positives;
- page quarantine after two invalid OCR attempts;
- query-relative labels for shared sections;
- all remaining same-page sections become hard negatives;
- deterministic page-grouped split with no leakage;
- byte-identical outputs on repeated runs;
- complete provenance and join integrity.

Run at minimum:

```bash
python -m pytest -q \
  tests/test_deepseek_grounding.py \
  tests/test_sciegqa_final_labeling.py \
  tests/test_sciegqa_parser_viewer.py \
  tests/test_sciegqa_4k_selection.py \
  tests/test_sciegqa_4k_extract.py \
  tests/test_sciegqa_4k_pipeline.py \
  tests/test_sciegqa_evidence_poc.py
```

The new test file must be added. Do not claim completion from unit tests alone;
run one real-page smoke, then a small multi-page shard smoke before submitting
the full array.

## Phase 10: final audits

`audits/labeling_audit.json` must report:

- source query/page counts;
- OCR-valid, retry-valid, and terminal-failure page counts;
- section counts by kind and domain;
- section member/word-count distributions for diagnostics only;
- pages and queries quarantined by each reason;
- retained page/query counts;
- positive and hard-negative pair counts;
- hard negatives per query distribution;
- queries per retained page distribution;
- proof that each retained query has exactly one positive;
- proof that no partial labels remain;
- proof that every negative is same-page;
- source/model/repository revisions and every input/output SHA-256.

`audits/split_audit.json` must report achieved 70/15/15 proportions, page and
query counts, pair counts, domain counts, query-intent counts, and maximum
stratification deviations.

`audits/determinism.json` must record two independent finalization runs over
the same merged OCR artifacts and prove byte-identical hashes for all section,
label, quarantine, split, and audit manifests.

## Stop conditions

Stop and report instead of guessing if:

- the SOL commit does not contain `546d9c8`;
- the source is not exactly 4,000 queries and 3,711 pages;
- source, page image, model, or prompt provenance differs;
- page files or recorded hashes are missing;
- sharded jobs would write shared mutable files;
- the model/environment cannot reproduce the one-page smoke;
- retry output would overwrite attempt-one artifacts;
- semantic-section output differs from the checked-in viewer golden test;
- a proposed implementation copies the viewer logic instead of sharing it;
- any retained query lacks exactly one positive;
- any partial relation survives quarantine;
- any page crosses splits;
- a design choice would require rendered crops, length-based splitting,
  cross-page negatives, negative sampling, or manual annotation;
- scratch capacity or cluster policy prevents a safe run.

## Completion criteria

This handoff is complete only when:

1. All 3,711 pages have one selected valid OCR attempt after at most two
   attempts, or a terminal quarantine record after exactly two failed or
   invalid attempts.
2. Semantic sections are generated with the shared verified viewer logic.
3. Every retained query has exactly one positive and all other same-page
   sections are hard negatives.
4. Every ambiguous or invalid page and all its queries are quarantined.
5. Page-grouped 70/15/15 split manifests pass leakage and stratification
   audits.
6. All manifests are deterministic, hashed, provenance-complete, and stored on
   scratch.
7. Focused tests, real-page smoke, multi-page shard smoke, merge validation,
   labeling validation, and split validation all pass.
8. `sol/CURRENT_SOL_TASK.md` contains the final job IDs, run path, results, next
   action, and blockers without stale history.

Do not start VLM training in this task. The deliverable is the validated,
query-relative evidence dataset and its splits.
