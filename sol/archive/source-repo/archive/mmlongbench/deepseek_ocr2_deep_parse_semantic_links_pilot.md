# DeepSeek-OCR-2 Deep-Parse Semantic-Link SOL Contract

Status: completed and superseded by the academic gold-page OCR-only task

Binding experiment design:
`docs/superpowers/specs/2026-07-17-deepseek-ocr2-deep-parse-semantic-links-design.md`

This file defines the SOL-specific execution and return contract. The binding
design defines the exact 13 pages, prompts, fidelity rules, inventories,
output schemas, and completion criteria. If this file and the design conflict,
stop and report the conflict rather than choosing silently.

## 1. Purpose

Run the approved preliminary prompt experiment on 13 selected difficult
MMLongBench-Doc pages. Compare every raw-page prompt arm in the design with:

1. a full-page grouping arm that also receives a compact inventory derived
   solely from the existing DeepSeek-OCR-2 output; and
2. a full-page grouping arm whose visual entries use mechanically compressed
   excerpts from the paper-faithful `Parse the figure.` outputs.

SOL performs research, minimal implementation, rendering/cropping, inference,
validation, and Git result return. SOL must not build or serve a webpage.

## 2. Required reading and independent research

Read, in order:

1. `AGENTS.md`
2. `SOLinstrucitons.md`
3. `sol/AGENTS.md`
4. this file
5. the binding experiment design
6. [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/pdf/2510.18234),
   especially Sections 4.3.1 and 4.3.3
7. [official DeepSeek-OCR repository](https://github.com/deepseek-ai/DeepSeek-OCR)
8. [official DeepSeek-OCR-2 model card](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
   and pinned remote inference code

Do independent primary-source research before editing or submitting. Record
the research in the returned `research.md`. Clearly distinguish published
behavior from experimental prompts and repository choices. Research may make
minimal syntax corrections to experimental prompts as allowed by the design;
it may not change the pages, arms, deep-parse prompt, or source-only context
rule.

## 3. Frozen code and model baseline

- Branch: `codex/mmlongbench-ocr2-313`
- Model: `deepseek-ai/DeepSeek-OCR-2`
- Revision: `aaa02f3811945a91062062994c5c4a3f4c0af2b0`
- Tokenizer: same pinned revision
- Base size: `1024`
- Image size: `768`
- Crop mode: `true`
- Evaluation mode: `true`
- Model dtype: BF16
- Existing runner: `scripts/document_parsing/deepseek_runner.py`
- Existing CLI: `scripts/document_parsing/run_deepseek_ocr.py`

Extend the existing runner minimally so each run can bind an exact prompt and
input kind. Do not create an unrelated inference implementation, substitute
vLLM, or change the pinned full-page baseline.

## 4. Inputs

The frozen 13-page selection is in the binding design and is resolved against:

```text
pilot_data/mmlongbench_ocr2_unstructured_313/page_plan.jsonl
sol_results/mmlongbench_ocr2_unstructured_313/20260717T182300Z/pages.jsonl
sol_results/mmlongbench_ocr2_unstructured_313/20260717T182300Z/deepseek_ocr2/
```

Three non-overlapping OCR-only inventory parts are prepared under:

```text
pilot_data/mmlongbench_ocr2_deep_parse_links/agent_inventories/
  part_a.jsonl
  part_b.jsonl
  part_c.jsonl
```

Together they contain exactly 13 pages, 336 atomic units, 47 visual units, and
25 mechanically truncated excerpts. They were checked against the returned
`segments.jsonl`, but they are not an independent source of truth. Implement a
deterministic regeneration/validation path from `grounded_output.txt` and its
mechanical parsed segments before using them as prompts.

## 5. OCR-only context rule

Assisted prompts may contain only material already emitted by
DeepSeek-OCR-2:

- source segment ID and OCR type;
- normalized OCR bounding box;
- whitespace-collapsed text from that same OCR unit; and
- mechanically truncated text from a later DeepSeek deep-parse response.

No page image inspection, PDF inspection, semantic-section output, viewer
output, external source, neighboring-caption association, paraphrase,
summary, or inferred visual description may enter an inventory.

Full-page OCR text is capped extractively at 240 Unicode characters: keep the
first 239 characters and append `…`. A visual with no OCR-authored description
is exactly `[image]` or `[figure]`, according to its OCR type.

Deep-parse text is capped extractively at 320 Unicode characters after
whitespace collapse. Preserve the complete raw response separately. Do not use
a language model to rewrite or summarize it. This makes context construction
reproducible with deterministic code or a smaller text-only model.

## 6. Paper-faithful deep parse

Deep parsing is a secondary call on every unique image/figure box from the
successful grounded full-page OCR. Crop that exact source-image region with no
added margin, annotation, OCR text, rescaling, or enhancement. Pass only the
crop and this exact prompt:

```text
<image>
Parse the figure.
```

Use the pinned official OCR-2 `model.infer` path and parameters in the binding
design. Do not add a system prompt, page context, coordinates, OCR transcript,
format request, or suffix. Preserve raw responses byte for byte. Record every
paper-unspecified crop choice as an experiment choice, not a paper claim.

## 7. Execution sequence

1. Validate the 13 selected page IDs, source PDFs, baseline page-image hashes,
   and successful baseline OCR records.
2. Complete and record the independent research gate.
3. Implement focused tests for prompt override provenance, inventory
   regeneration, extractive truncation, visual placeholders, materialized
   prompt hashing, and raw-output preservation.
4. Freeze the exact prompt-arm manifest from the binding design.
5. Regenerate the compact inventories and prove they match the three prepared
   inventory parts in source IDs, order, boxes, types, excerpts, and visual
   placeholders.
6. Re-render selected pages using the existing renderer and require the known
   baseline page-image hashes before inference or cropping.
7. Run the two-page smoke defined in the design. Semantic quality is not a
   smoke gate; a returned raw string and intact provenance are sufficient.
8. Run every raw-image full-page arm plus the OCR-inventory-assisted arm on all
   13 pages.
9. Run the exact paper-faithful deep-parse call on every unique detected visual
   region.
10. Materialize deterministic deep-parse excerpts and run the
    deep-parse-assisted grouping arm on all 13 pages.
11. Package text/JSON results, audit them, commit the scoped result, and push
    this branch.

An inference crash may be retried once with identical inputs. Poor semantic
output is a result and must not be retried, rewritten, or repaired. A parse
failure must retain its raw response and may not block unrelated arms.

## 8. Result and Git boundary

Return the exact bundle defined in the binding design under:

```text
sol_results/mmlongbench_ocr2_deep_parse_links/<RUN_ID>/
```

The commit may contain only small source code/tests needed for the run, the
three prepared text/JSON inventory inputs or their deterministic canonical
merge, small result text/JSON, `sol/CURRENT_SOL_TASK.md`, and `sol/sol_logs.md`.

Do not commit PDFs, rendered pages, crops, model files, caches, environments,
Slurm stdout/stderr, or a website. Keep compute artifacts on scratch. Every
completed run must preserve its exact materialized prompt, input hash, output
hash, model revision, and inference parameters.

Before committing, prove the requested run inventory and result artifact set
match exactly, all hashes verify, source-unit identities match across assisted
arms, and repeated visual IDs across groups were not collapsed. Push only
`codex/mmlongbench-ocr2-313`; never push `main`.

## 9. Stop conditions

Stop and update `sol/CURRENT_SOL_TASK.md` when:

- a primary source contradicts the binding design;
- the pinned model cannot accept an exact required prompt through its official
  inference path;
- selected page or OCR hashes differ;
- inventory regeneration requires non-OCR information;
- a crop cannot be reproduced from recorded OCR coordinates;
- a job requires changing the model, revision, or execution backend; or
- Git authentication prevents the final push.

Do not broaden scope or silently substitute a different methodology.
