# MMLongBench Gold Pages: DeepSeek-OCR-2 SOL Contract

Status: prepared, not yet launched

This is the binding first-stage SOL handoff for the gold-page quadrant-reranker
experiment. It ends after OCR acquisition and Git-safe result packaging.

## 1. Objective and hard scope

Run the repository's pinned DeepSeek-OCR-2 full-page grounded-Markdown OCR on
every unique valid official gold page required by the answerable questions in
the full MMLongBench-Doc dataset at:

```text
/scratch/lmalveau/agenticdocai/data/MMLongBench-Doc
```

Build and return the frozen question/page mappings required to consume those
OCR results locally.

Do not:

- run semantic-section construction or hierarchical segmentation;
- run the Qwen reranker or any answerer;
- create halves, quadrants, image-caption pairs, or evaluation candidates;
- use questions, answers, or evidence annotations as OCR-model input;
- evaluate OCR quality against answers or repair poor OCR;
- build or serve a website; or
- add a prompt arm, deep-parse call, model, or inference backend.

Poor OCR is data. Preserve the raw result and diagnostics rather than silently
repairing or excluding it.

## 2. Required first inspection

Before writing adapters or submitting a job, read:

1. `AGENTS.md`
2. `docs/SOL_INSTRUCTIONS.md`
3. `sol/AGENTS.md`
4. `sol/CURRENT_SOL_TASK.md`
5. this contract
6. `scripts/document_parsing/deepseek_runner.py`
7. `scripts/document_parsing/run_deepseek_ocr.py`
8. `scripts/mmlongbench/pilot.py`
9. `scripts/mmlongbench/prepare_pilot.py`
10. `sol/archive/jobs/mmlongbench_completed/`
11. `sol/archive/task_specs/mmlongbench_academic_gold_deepseek_ocr2.md`
12. `tests/test_mmlongbench_ocr2_pilot.py`
13. `tests/test_mmlongbench_ocr2_sol.py`

Inspect the full dataset root to identify its authoritative table, document
directory, field schema, page-number convention, and source snapshot metadata.
Reuse the existing renderer, sharded OCR runner, retry merger, and audited
packager where practical. Generalize them through a narrow adapter; do not
build a second OCR pipeline merely because the earlier run used a different
manifest.

The archived academic contract is procedural precedent only. Its 26-document,
150-question, and 193-page counts and its compact PDFs are retired and must not
be reused as the new inventory.

## 3. Git and execution boundary

Start from the exact pushed `main` commit containing this handoff. Create and
push only:

```text
codex/mmlongbench-gold-pages-ocr2
```

Record the base and execution commit hashes in the run manifest. Keep large
inputs, rendered pages, model caches, environments, and logs on scratch. Never
run rendering or inference on the login node.

Do not merge into `main`. Do not alter or delete the retained unstructured
313-page result.

## 4. Frozen selection contract

Discover the dataset schema, then create a deterministic, validated inventory
with these Git-safe files:

```text
pilot_data/mmlongbench_ocr2_gold_pages/
  dataset_snapshot.json
  documents.jsonl
  questions.jsonl
  question_page_links.jsonl
  page_plan.jsonl
  exclusions.jsonl
  selection_summary.json
  SHA256SUMS
```

Selection rules:

1. Include every answerable question whose official evidence-page list is
   nonempty and resolves to valid pages in its source PDF.
2. Record each unique source document/page once in `page_plan.jsonl`.
3. Preserve every question-to-page relationship separately in
   `question_page_links.jsonl`.
4. Record unanswerable rows, empty evidence lists, invalid page references,
   missing documents, duplicate question IDs, and other rejected rows in
   `exclusions.jsonl` with explicit reason codes.
5. Preserve the dataset's original evidence-page values and also record the
   resolved one-based PDF page number. Verify the dataset convention from its
   own metadata/code instead of guessing or copying the retired subset's
   convention.
6. Build stable page IDs from immutable source identity, including the source
   document hash and resolved page number. The inventory must be identical
   across input ordering.
7. Freeze hashes for the authoritative dataset table, every referenced PDF,
   every generated manifest, and the source snapshot/revision information that
   is actually available.
8. Compute and record observed question, document, unique-page, link, and
   exclusion counts before any OCR job. Stop if a second preparation pass
   produces different identities, counts, or hashes.

Do not copy full PDFs or rendered PNGs into Git. The full dataset root remains
the rendering source on SOL.

## 5. Page rendering contract

Render only the unique rows in `page_plan.jsonl` with the existing PyMuPDF
renderer at matrix scale `2.0`. Keep `page_plan.jsonl` as the pre-render
inventory and write the observed render metadata to scratch `pages.jsonl`.
Never add PNG paths or hashes retroactively to `page_plan.jsonl`.

Each `pages.jsonl` row must retain:

- stable page ID;
- document ID, source filename, and document SHA-256;
- original evidence-page value and resolved one-based page number;
- rendered width, height, matrix scale, PyMuPDF version, and PNG SHA-256; and
- source dataset snapshot identity.

Fail closed on a missing PDF, invalid page index, duplicate page ID, document
hash mismatch, or render destination collision.

## 6. Pinned OCR inference

Use the existing repository runner and this exact configuration:

- model/tokenizer: `deepseek-ai/DeepSeek-OCR-2`;
- revision: `aaa02f3811945a91062062994c5c4a3f4c0af2b0`;
- prompt: `<image>\n<|grounding|>Convert the document to markdown.`;
- base size: `1024`;
- image size: `768`;
- crop mode: `true`;
- evaluation mode: `true`;
- `save_results`: `false`;
- dtype: BF16 through the existing pinned SOL environment; and
- maximum generated tokens: `8192`.

Do not change prompt whitespace, provide the benchmark question or answer,
substitute vLLM, or update the model revision.

## 7. Required implementation and tests

Prefer a small MMLongBench gold-page inventory adapter plus thin active SBATCH
wrappers around the existing render, OCR, retry, merge, and packaging code.
Follow repository naming and interfaces discovered during the required first
inspection.

Before the full run, add focused tests proving:

- deterministic question/page selection under reversed input order;
- correct handling of single-page and cross-page links;
- verified page-number conversion and PDF bounds;
- stable page IDs and duplicate rejection;
- explicit exclusion reason codes;
- questions and answers never enter OCR prompts or page manifests;
- pre-render `page_plan.jsonl` remains distinct from rendered `pages.jsonl`;
- the pinned model, revision, prompt, and inference parameters are unchanged;
- the sharded job covers each page exactly once; and
- packaging rejects missing references, hash mismatches, and forbidden files.

Run focused tests only. Do not run unrelated long suites on SOL.

## 8. Execution sequence

1. Perform lightweight repository and dataset discovery.
2. Implement and test the minimal inventory adapter and active wrappers.
3. Generate the inventory twice and prove deterministic equality.
4. Freeze the observed counts and all selection hashes in
   `selection_summary.json` and `SHA256SUMS`.
5. Render the complete unique gold-page inventory once on a CPU compute
   allocation.
6. Select a deterministic smoke set from the frozen inventory containing pages
   from distinct documents and, where dataset metadata permits, distinct
   evidence-source categories.
7. Run the smoke set through the pinned OCR path. Gate only execution,
   provenance, raw-output preservation, and reference/hash validity; semantic
   OCR quality is not a gate.
8. Submit the full sharded GPU run with shard count/concurrency derived from the
   frozen page count and available SOL policy.
9. Retry only infrastructure/crash failures once using byte-identical images
   and identical inference parameters. Do not retry for poor text, repetition,
   truncation, empty parsed units, or other semantic outcomes.
10. Merge, package, audit, checksum, commit, and push the Git-safe result bundle
    to the task branch.

After submission, confirm that jobs start successfully and then stop polling
unless monitoring is explicitly requested.

## 9. Scratch and returned result contract

Use a run root such as:

```text
/scratch/lmalveau/mmlongbench_ocr2_gold_pages/runs/<RUN_ID>/
```

Return the Git-safe result under:

```text
sol_results/mmlongbench_ocr2_gold_pages/<RUN_ID>/
```

The returned bundle must include:

- the frozen dataset/selection summary, question records with their benchmark
  gold-answer metadata, and question/page mappings;
- successful `pages.jsonl` with render and source provenance;
- `deepseek_ocr2/runs.jsonl`;
- `deepseek_ocr2/segments.jsonl` with parsed grounded atomic units;
- byte-preserved `grounded_output.txt`, deterministic `document.md`,
  `parse_diagnostics.json`, and `run.json` for every successful page;
- runtime, environment, GPU, renderer, Git, model, and prompt provenance;
- an explicit terminal-failure/completion audit;
- `artifact_manifest.json`; and
- `SHA256SUMS` covering every returned file except the checksum file itself.

Do not return PDFs, PNGs, model weights, caches, environments, Slurm logs,
semantic sections, quadrants, reranker output, generated model answers,
evaluation output, or a website. The benchmark gold-answer metadata in the
frozen question manifest is required provenance, but it must never be passed to
the OCR model. Do not omit a poor raw response because parsing failed.

Use Git for the result handoff. If the complete text/JSON bundle cannot be
pushed safely, stop and report the exact size or Git failure rather than
dropping artifacts, changing transport, or committing binaries.

## 10. Completion checks

Before pushing, prove:

- selected question/page/link/exclusion counts match the frozen summary;
- every unique gold page is rendered and has exactly one successful run or one
  explicit terminal failure;
- every cross-page question retains all of its valid page links;
- no excluded or nongold page entered the OCR inventory;
- all run records use the exact pinned OCR configuration;
- every raw, Markdown, diagnostics, run, page, and segment reference resolves
  and matches its hash;
- all final checksums verify;
- forbidden files and downstream experiment artifacts are absent; and
- a fresh clone of the pushed branch can validate the result bundle without
  access to scratch-rendered PNGs.

Update `sol/CURRENT_SOL_TASK.md` with the exact branch, execution commit, frozen
counts, run ID, job IDs, result path, latest state, and any blocker. Keep logs
out of the handoff.

## 11. Stop conditions

Stop and report instead of broadening scope if:

- the dataset's authoritative table or page-number convention cannot be
  established;
- source PDFs or required fields are missing;
- deterministic preparation produces different inventories;
- the pinned environment, model revision, or runner is unavailable;
- a prompt, model, backend, retry-policy, or selection-rule change appears
  necessary;
- packaging would require committing PDFs or rendered page images; or
- Git authentication or size limits prevent the complete branch-only push.
