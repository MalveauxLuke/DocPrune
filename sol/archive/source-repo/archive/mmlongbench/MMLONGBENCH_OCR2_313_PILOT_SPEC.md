# MMLongBench-Doc DeepSeek-OCR-2 Off-Domain Segmentation Pilot

Status: active SOL run; explicitly reduced to an incomplete exploratory subset

Design date: 2026-07-16

Execution boundary: SOL performs rendering and DeepSeek-OCR-2 inference only;
the local workstation performs semantic segmentation and webpage construction
after SOL pushes the completed OCR artifacts.

## 1. Goal

Run the pinned DeepSeek-OCR-2 configuration over every page of ten
non-academic MMLongBench-Doc PDFs, then return the complete OCR artifact set to
this repository through Git. The pilot is a preliminary qualitative stress test
of the existing academic-document semantic-segmentation rules on difficult,
off-domain document types such as manuals, guidebooks, brochures, device
instructions, and mixed text/figure layouts.

The fixed input size is:

```text
documents = 10
pages     = 313
```

Every selected PDF must contain at most 50 pages. The PDFs must not be
truncated, and research papers must not be substituted into the sample.

## 2. Scope boundaries

### SOL is authorized to

1. Validate the tracked PDF and manifest hashes.
2. Render all 313 PDF pages into deterministic page images on scratch.
3. Run the pinned DeepSeek-OCR-2 inference, including the existing conservative
   validation and one-retry policy.
4. Preserve raw grounded output, atomic parsed blocks, diagnostics, and run
   provenance.
5. Validate and commit the completed OCR artifact bundle.
6. Push one scoped result commit to the prepared pilot branch.

### SOL must not

- construct semantic sections;
- apply query-relative positive, partial, or negative labels;
- build a viewer manifest or webpage;
- run a web server or attempt visual display;
- add rendered page PNGs, model weights, caches, Slurm logs, or environments to
  Git;
- alter the segmentation rules to accommodate off-domain pages; or
- touch the existing SciEGQA test split or segment-reranker experiment.

### Local work after the result push

After the result commit is pulled locally, a separate local task will:

1. Verify the returned inventory and hashes.
2. Reparse the authoritative raw grounded outputs.
3. Apply `build_deepseek_semantic_sections` without off-domain special cases.
4. Record structural and failure diagnostics.
5. Build and serve a precomputed static review webpage locally.

That later webpage phase is not part of the SOL execution task.

## 3. Frozen source snapshot

Source dataset: `yubo2333/MMLongBench-Doc`

Repository revision: `2ff6aa9237fc777b6627dc57a486e9225ac5fb86`

Source parquet: `data/train-00000-of-00001.parquet`

Source parquet SHA-256:
`bcdac3c96669634c34184814cede4fe57cf7ac0f98dde0e85936394f6a56a02d`

The selected source PDFs are currently available locally under:

```text
/Users/god/Documents/document vqa/MMLongBench-Doc/documents/
```

Implementation will copy only the ten approved PDFs and their associated
MMLongBench rows into a dedicated tracked pilot-data directory. The source
snapshot is immutable after the manifest is committed.

## 4. Approved document inventory

| Document | Type used by this pilot | Pages | SHA-256 |
|---|---|---:|---|
| `91521110100M_4K_UHD_Display_User_Manual_V1.1.pdf` | display user manual | 40 | `a3dfbac95e5c99ea9aa9fb830bd5f338d3278c7df28a775bc314acedc22b20d5` |
| `owners-manual-2170416.pdf` | refrigerator owner's manual | 32 | `91e70aaf49192b4deb640154a50d0b14976c831be11a7f858f9e953a761b8681` |
| `mi_phone.pdf` | mobile-phone guide | 30 | `16ee1cfbd58f9c6a3793c66ff5f4ec38dd864ac9fd7a9b7186f8af7db53f870b` |
| `watch_d.pdf` | smartwatch guide | 27 | `bb5fd3576ac080c867f200ddc90311cdb351fb8240c70c9468dc433936c136e6` |
| `ISEP_student_handbook_2020.pdf` | student handbook | 24 | `27163e80a71d12f267592d87c1112413ac85ed432c766f45e9558a47e6f9cb80` |
| `camry_ebrochure.pdf` | automotive brochure | 26 | `d3f8049bc0c48b735b7c4e0d6399ed9e8188afd332f58829f44e1c39cfb13528` |
| `honor_watch_gs_pro.pdf` | smartwatch guide | 42 | `2d5559e514dc54f616ffea8d8e054117e32da672bdbf9797883dd815fa990ee5` |
| `StudentSupport_Guidebook.pdf` | student-support guidebook | 44 | `7515cd264f0a33a782169d938a291e9ec4fc81a094a30a4ecf951d4cfa7ff336` |
| `GPL-Graduate-Studies-Professional-Learning-Brochure-Jul-2021.pdf` | graduate-studies brochure | 17 | `83424ce04cc38c567e121a1f6827e370a878917a8d8b5b7743e05f1c49042f5b` |
| `NUS-FASS-Graduate-Guidebook-2021-small.pdf` | graduate guidebook | 31 | `8e7802bfbb70e4126bcc6a5f4d06bcc1c549dd24cf4c90a143eab8971abe66a8` |
| **Total** |  | **313** |  |

The preparation validator must fail if any hash or page count differs, if the
sum is not exactly 313, or if any selected document exceeds 50 pages.

## 5. Repository layout

The implementation must use these tracked input and returned-output paths:

```text
pilot_data/mmlongbench_ocr2_unstructured_313/
  documents/                 # the ten immutable tracked PDFs
  documents.jsonl            # document IDs, page counts, bytes, and hashes
  questions.jsonl            # all released MMLongBench rows for these PDFs
  page_plan.jsonl            # exactly 313 ordered source-page records
  source_snapshot.json       # dataset revision and parquet provenance

sol_results/mmlongbench_ocr2_unstructured_313/<RUN_ID>/
  pages.jsonl                # SOL render paths, dimensions, versions, and hashes
  completion_audit.json
  artifact_manifest.json
  SHA256SUMS
  deepseek_ocr2/
    runs.jsonl
    segments.jsonl
    environment_freeze.txt
    raw/<PAGE_ID>/
      grounded_output.txt
      document.md
      parse_diagnostics.json
      run.json
```

Generated page images remain on SOL scratch and are not committed. The local
viewer will render its own display images from the identical tracked PDFs and
use normalized DeepSeek coordinates.

## 6. Page identity and rendering

Page IDs must be deterministic and collision-safe. They must bind the source
document SHA-256 and the 1-based PDF page number, rather than relying only on a
filename. The tracked `page_plan.jsonl` must contain one row for every page from
1 through the validated PDF page count, ordered first by the document order in
Section 4 and then by page number.

Each tracked page-plan row must record at least:

- stable `page_id`;
- source document filename and SHA-256;
- 1-based PDF page number;
- expected document page count.

After rendering on SOL, write a result `pages.jsonl` that preserves those
source fields and adds:

- rendered image path used on SOL;
- rendered PNG SHA-256 after rendering;
- render command and renderer version; and
- rendered width and height.

Rendering occurs on compute resources, not the login node. A render failure or
page-count mismatch blocks inference.

## 7. Pinned DeepSeek-OCR-2 inference

The inference contract remains unchanged:

- Model: `deepseek-ai/DeepSeek-OCR-2`
- Revision: `aaa02f3811945a91062062994c5c4a3f4c0af2b0`
- Prompt: `<image>\n<|grounding|>Convert the document to markdown.`
- Base size: `1024`
- Image size: `768`
- Crop mode: enabled
- Evaluation mode: enabled
- Maximum new tokens: `8192`

Reuse `scripts/document_parsing/deepseek_runner.py` and
`scripts/document_parsing/run_deepseek_ocr.py`; do not create a second inference
implementation. A new wrapper or manifest adapter may be added for this pilot.

Run a one-page smoke test before the full submission. The full run may be
sharded, but the returned bundle must merge to exactly one run row per page and
must reject duplicate or missing page IDs.

## 8. Failure and retry policy

Use the existing conservative inference policy. An output that is truncated,
malformed, repetitive, incomplete, or otherwise invalid receives one retry.
The second attempt must use the same pinned model, prompt, and inference
parameters while recording `attempt_index=2`.

Unlike the SciEGQA dataset pipeline, this preliminary viewer pilot must not
silently quarantine pages and declare success. After the retry pass:

- retain both attempt records and diagnostics;
- mark unresolved pages failed in the audit;
- do not push a result commit described as complete unless all 313 pages have a
  valid completed artifact; and
- report the exact blocking page IDs in `sol/CURRENT_SOL_TASK.md` if the run
  cannot complete.

Do not manually edit raw OCR output to make a page pass.

### 8.1 Explicit exploratory-subset override (2026-07-17)

The user explicitly overrode the retry and complete-313-page requirements for
this basic exploratory run after attempt 1. Package only the 306 pages that
passed the conservative attempt-1 quality gate. The bundle must be labeled
incomplete and must preserve the original 313-page source/render inventory plus
an exclusion audit for these seven pages:

- `mmlongbench_page_6a48afd2d59ff9b0debd`: `parse_diagnostics`, `repetition_detected`
- `mmlongbench_page_9872518950724cbaea59`: `likely_truncated`, `unbalanced_grounding_tags`
- `mmlongbench_page_acce2b0cceff4c736dab`: `parse_diagnostics`
- `mmlongbench_page_c0c87764f9dcce381905`: `likely_truncated`, `parse_diagnostics`, `unbalanced_grounding_tags`
- `mmlongbench_page_e8a888f2e189530a4815`: `parse_diagnostics`
- `mmlongbench_page_efce297bdc7b62ac7394`: `likely_truncated`, `unbalanced_grounding_tags`
- `mmlongbench_page_f1f9fc92a1269a3db891`: `generated_token_limit`, `likely_truncated`, `parse_diagnostics`

This override is limited to this run. Default merge and packaging behavior must
remain fail-closed for future runs unless an equally explicit incomplete-run
option is supplied.

## 9. Git result-return contract

Use a dedicated pilot branch with the `codex/` prefix. SOL pulls the prepared
branch containing the PDFs, manifests, code, and current-task handoff. After
inference, SOL stages only:

```text
sol_results/mmlongbench_ocr2_unstructured_313/<RUN_ID>/
sol/CURRENT_SOL_TASK.md
sol/sol_logs.md
```

Before committing this explicitly incomplete exploratory bundle, SOL must prove:

1. `runs.jsonl` has exactly 306 unique completed page rows.
2. The run page-ID set exactly equals the returned `pages.jsonl`; the completion
   audit records the original 313-page count and exactly seven excluded page IDs.
3. Every raw artifact named in a run row exists and matches its recorded hash.
4. `segments.jsonl` contains no duplicate segment IDs and references only the
   306 packaged page IDs.
5. `artifact_manifest.json` inventories every committed result payload except
   itself and `SHA256SUMS`, which avoids a self-referential hash.
6. `SHA256SUMS` covers every committed result file except itself and verifies
   successfully from the run directory.
7. No PNG, PDF, model, cache, environment directory, or Slurm output file is
   staged in the result commit.
8. `git diff --cached --stat` contains only the authorized paths above.

SOL then creates one scoped result commit and pushes the pilot branch. It must
record the pushed commit SHA in `sol/CURRENT_SOL_TASK.md`. If authentication or
remote rejection prevents the push, stop and report the local commit SHA; do
not switch remotes, rewrite history, or push to `main`.

Deleting the result directory in a later commit will not remove its objects
from Git history. Reclaiming repository size would require an explicitly
authorized history rewrite. This tradeoff is accepted for the preliminary
text/JSON OCR bundle, but rendered images remain excluded.

The selected source PDFs also remain subject to their original licensing terms.
Do not change repository visibility or publish them outside the repository's
existing access boundary as part of this task.

## 10. Local semantic-segmentation and viewer contract

The later local phase treats each returned `grounded_output.txt` as
authoritative. It reparses atomic blocks and passes them to the existing
`build_deepseek_semantic_sections` implementation. No JavaScript code may
recompute segmentation.

The academic segmentation rules are intentionally left unchanged so the pilot
can reveal off-domain failure modes. The review surface must make at least the
following visible per page:

- original rendered page;
- atomic DeepSeek blocks and raw types;
- resulting semantic sections and section types;
- member-box unions and reading order;
- caption primary/fallback link provenance;
- malformed, ungrounded, standalone-visual, and orphan-title diagnostics; and
- authoritative raw OCR and derived Markdown.

Because MMLongBench-Doc provides page-level evidence rather than section-level
ground-truth regions, this pilot must not report semantic-segmentation accuracy,
precision, recall, or a single aggregate quality score. It may report objective
operational and structural counts, while segmentation quality remains a visual
audit.

## 11. Active-task replacement and archive policy

When implementation begins, archive the superseded segment-evidence task state
under a dated subdirectory of `sol/archive/colqwen/`. This includes:

- the previous `sol/CURRENT_SOL_TASK.md` content;
- `sol/sol_stage0_stage1_agent_handoff.md`;
- the two superseded task specifications; and
- all tracked `.bak-*` files currently under `sol/` and `sol/task_spec/`.

Do not delete experiment code, Slurm wrappers, `sol/sol_logs.md`, or durable
module context. After archiving, `sol/CURRENT_SOL_TASK.md` must be a short
handoff pointing to this specification as the sole binding active task.

## 12. Completion criteria

Preparation is complete when:

- the ten PDFs and frozen manifests are tracked on the pilot branch;
- focused tests prove the ten-document, 313-page, at-most-50-page contract;
- the SOL wrapper and artifact-audit path are tested without downloading model
  weights locally;
- the superseded task documents are archived;
- `sol/CURRENT_SOL_TASK.md` contains only the new pilot state and next action;
  and
- the prepared branch is available for SOL.

SOL execution is complete when the pushed result commit satisfies every gate in
Section 9 as amended by Section 8.1. Website construction begins only after
that commit is pulled and verified on the local workstation.
