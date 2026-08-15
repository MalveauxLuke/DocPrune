# MMLongBench Academic Gold Pages: DeepSeek-OCR-2 SOL Contract

Status: active

This is the binding SOL execution and return contract. It deliberately ends
after OCR acquisition. The local user will inspect the returned OCR and decide
what the correct semantic segments should be before any segmentation or VQA
experiment is authorized.

## 1. Objective and hard scope

Run the repository's pinned DeepSeek-OCR-2 full-page grounded Markdown OCR on
all 193 locally selected MMLongBench-Doc academic gold pages. Preserve the raw
model response, deterministic Markdown conversion, parsed atomic grounded
units, diagnostics, and complete provenance.

Do not:

- run the repository's semantic-section heuristics;
- generate candidate or gold semantic segments;
- run Qwen or any document-VQA scoring;
- inspect answers to alter OCR, page selection, retries, or packaging;
- build or serve a website; or
- add another model, inference backend, prompt arm, or deep-parse call.

Poor OCR is an experimental result. Preserve it rather than repairing it.

## 2. Frozen branch and local corpus

- Branch: `codex/repo-consolidation`
- Corpus root: `pilot_data/mmlongbench_ocr2_academic_gold/`
- MMLongBench-Doc revision:
  `2ff6aa9237fc777b6627dc57a486e9225ac5fb86`
- Source parquet SHA-256:
  `bcdac3c96669634c34184814cede4fe57cf7ac0f98dde0e85936394f6a56a02d`
- `page_plan.jsonl` SHA-256:
  `f95777a81cfdb6de2acd8a9dfdad6df072016fa50da3114f71c1d69fdc562c15`
- `questions.jsonl` SHA-256:
  `df493dec7b673a47c086bbad0d6c630bca06b5b3d9d0fbb40bc180e323675238`
- `SHA256SUMS` SHA-256:
  `bd458b20d371af5d17c78b743201035b5cb868f55d5f57942eed821ad8f2a0e7`

Frozen counts:

| Artifact | Count |
|---|---:|
| Academic source documents | 26 |
| Retained answerable questions | 150 |
| Unique gold pages | 193 |
| Question-page links | 260 |
| Excluded academic rows | 54 |

The 54 exclusions are explicit in `exclusions.jsonl`: 50 unanswerable rows and
four rows whose official evidence-page lists are empty or contain invalid page
numbers. They were excluded, not repaired. The 26 tracked PDFs contain only
the selected gold pages and total approximately 67 MiB. They are the complete
SOL PDF input; do not fetch or reconstruct the original full documents.

Before rendering, run:

```bash
python scripts/mmlongbench/prepare_academic_gold.py validate \
  --output-root pilot_data/mmlongbench_ocr2_academic_gold
```

Also verify every entry in the corpus `SHA256SUMS`. Stop if any count, hash,
identifier, PDF page count, or original-to-subset page mapping differs.

## 3. Page interpretation

`page_plan.jsonl` is the sole page inventory. Each row contains:

- stable `page_id`;
- `source_document_filename` and original 1-based `source_page_number`;
- `subset_document_filename` and compact-PDF 1-based
  `subset_page_number`;
- source and subset document hashes; and
- linked question IDs for provenance only.

Render `subset_page_number` from `documents/<subset_document_filename>` and
carry the original source-document/page fields unchanged into every rendered
page and returned record. Never use `source_page_number` as the compact PDF
index.

Adapt the existing renderer minimally to accept these field names and explicit
expected counts of 26 documents and 193 pages. Keep a focused regression test
for the mapping. Do not create a parallel rendering implementation.

## 4. Frozen OCR inference

Use the existing repository runner:
`scripts/document_parsing/deepseek_runner.py` and CLI
`scripts/document_parsing/run_deepseek_ocr.py`.

The exact configuration is:

- model and tokenizer: `deepseek-ai/DeepSeek-OCR-2`;
- revision: `aaa02f3811945a91062062994c5c4a3f4c0af2b0`;
- prompt: `<image>\n<|grounding|>Convert the document to markdown.`;
- base size: `1024`;
- image size: `768`;
- crop mode: `true`;
- evaluation mode: `true`;
- `save_results`: `false`;
- dtype: BF16 through the existing pinned runner; and
- maximum generated tokens: `8192`.

Do not change whitespace in the prompt, add a system prompt, provide the
question or answer to the OCR model, or substitute vLLM or another model.
Render at the existing `2.0` PyMuPDF matrix scale and record the exact renderer
version, dimensions, image hash, and render scale.

## 5. Execution sequence

1. Validate the Git branch, corpus hashes, manifests, PDF counts, and 193 page
   mappings on a login node using only lightweight checks.
2. On an appropriate compute allocation, add focused tests and the minimal
   adapter needed to render/package this 26-document, 193-page corpus using the
   existing pilot pipeline.
3. Render all pages once to scratch. Keep rendered PNGs out of Git.
4. Run this exact four-page smoke through the pinned OCR path:

   | Coverage | Page ID | Compact PDF page | Original page |
   |---|---|---|---:|
   | Figure | `mmlongbench_page_65dd33dc38e0d596dbec` | `2005.12872v3.gold_pages.pdf`, page 3 | 22 |
   | Table | `mmlongbench_page_48272f993e0e9facc270` | `2005.12872v3.gold_pages.pdf`, page 2 | 13 |
   | Chart | `mmlongbench_page_a5c1661e910d8023390f` | `2023.acl-long.386.gold_pages.pdf`, page 4 | 7 |
   | Plain text | `mmlongbench_page_d062e88e25f4a943f409` | `2310.05634v2.gold_pages.pdf`, page 2 | 2 |

5. The smoke gate checks execution and provenance only: each page must return
   a raw string, its files must hash correctly, and its parsed representation
   must either succeed or preserve explicit diagnostics. Semantic quality is
   not a gate.
6. Run all 193 pages. A crash or infrastructure failure may be retried once
   with byte-identical image input and identical inference parameters. Do not
   retry for poor OCR, repetition, truncation, empty atomic-unit parsing, or
   other semantic quality outcomes.
7. Package the successful outputs and any terminal failures using the existing
   audited result packager. Generalize only its expected document/page counts
   and page-field adapter where needed.
8. Validate the final inventory, all references, hashes, and forbidden-file
   rules. Commit the scoped implementation/results and push this same branch.

## 6. Scratch and returned result contract

Use scratch for renders, caches, model files, Slurm logs, and intermediate run
state. Recommended run root:

```text
/scratch/lmalveau/mmlongbench_ocr2_academic_gold/runs/<RUN_ID>/
```

Return the Git-safe bundle under:

```text
sol_results/mmlongbench_ocr2_academic_gold/<RUN_ID>/
```

The bundle must include:

- selected successful `pages.jsonl` with original/subset page provenance;
- `deepseek_ocr2/runs.jsonl`;
- `deepseek_ocr2/segments.jsonl` containing parsed atomic OCR boxes/units;
- for every successful page, byte-preserved `grounded_output.txt`, derived
  `document.md`, `parse_diagnostics.json`, and `run.json`;
- environment freeze and runtime/model provenance;
- completion/terminal-failure audit;
- artifact manifest and `SHA256SUMS`; and
- only small code/tests needed for reproducibility.

The returned bundle must not contain PDFs, PNGs, page crops, model/cache files,
environments, Slurm stdout/stderr, semantic-section output, Qwen output, or a
website. Do not omit a bad raw output merely because atomic parsing fails.

## 7. Completion checks

Before pushing, prove:

- the input plan has exactly 193 unique page IDs across 26 compact PDFs;
- each rendered page uses its recorded `subset_page_number` and retains its
  original `source_page_number`;
- every selected completed run has the exact model revision, prompt hash,
  inference parameters, image hash, and attempt index;
- raw, Markdown, diagnostics, run, segment, and environment references resolve
  and match their recorded hashes;
- page/run/segment IDs are nonempty and unique where required;
- terminal failures, if any, are explicitly enumerated and match the missing
  successful page IDs;
- all final checksums verify; and
- no forbidden binary or downstream-experiment artifacts are present.

Run focused corpus, grounding, renderer, runner, retry, and packaging tests.
Do not run unrelated long test suites.

## 8. Stop conditions

Stop and update `sol/CURRENT_SOL_TASK.md` if:

- the tracked corpus does not match its frozen counts or hashes;
- a compact PDF cannot reproduce its recorded page mapping;
- the pinned model/revision or existing runner is unavailable;
- completing the task would require a prompt, model, backend, input-page, or
  retry-policy change;
- packaging would require committing rendered images/PDFs; or
- Git authentication prevents the branch-only push.

Do not broaden scope or silently substitute a different methodology.
