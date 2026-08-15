# Segment Reranker Corpus Architecture Specification

Status: approved design, implementation not started
Design date: 2026-07-15
Primary output: a self-contained, versioned corpus built from DUDE, SlideVQA, and TAT-DQA, with MMLongBench-Doc held out for evaluation
Downstream consumer: the segment-evidence classifier/reranker defined in `sol/task_spec/current_design_architecture_plan.md`

This document is the authoritative architecture for gathering and normalizing the multi-dataset corpus. It is written for an implementing agent with no prior conversation context. The agent must verify live source state before downloading anything, follow the pinned-revision rules below, and fail closed when released data do not satisfy the asserted schemas or counts.

The current scope ends after immutable assets, canonical tables, difficulty-set manifests, splits, and audits have been produced. Planner decomposition, evidence-region generation, negative mining, LoRA training, and human MMLongBench region annotation are downstream phases with interfaces defined here but no implementation authorization from this specification alone.

---

## 1. Goal

Build a large document-QA corpus for later LoRA training of a roughly 2B-parameter VLM segment reranker.

The deployed system is:

```text
long document
  -> ColPali retrieves approximately five pages
  -> a planner decomposes the original question
  -> retrieved pages are divided into semantic segments
  -> a VLM reranker scores each atomic-query/segment pair
```

The downstream model estimates:

```text
P(candidate segment supports atomic query)
```

It is not trained here to search the complete PDF, calculate the final answer, or execute the original multi-hop reasoning graph.

### 1.1 Why this corpus exists

No selected source supplies all required supervision uniformly:

- DUDE supplies broad, realistic, human-authored questions and human answer-location boxes for a subset.
- SlideVQA supplies evidence-slide labels and dense structural boxes over charts, diagrams, figures, tables, and slide text.
- TAT-DQA supplies dense financial table/text questions, derivations, supporting facts, and mappings to OCR blocks.
- MMLongBench-Doc supplies the final long-document transfer benchmark but only page-level evidence and no train split suitable for this project.

The corpus therefore preserves source-specific annotation meaning instead of flattening every source into an allegedly equivalent “gold evidence” label.

### 1.2 Success condition for this phase

This phase is complete only when a fresh machine can:

1. Reconstruct the exact downloaded source snapshots from provenance metadata.
2. Verify every original asset by SHA-256.
3. Resolve every canonical document, page, question, and native anchor.
4. Reproduce the source counts and split counts reported in the audits.
5. Reproduce all difficulty-set manifests deterministically.
6. Prove that no known MMLongBench source document is admitted to a training manifest.
7. Read the generated corpus without accessing network resources.

---

## 2. Explicit non-goals

Do not implement these while executing the gathering specification:

- Evidence-box generation with an MLLM.
- Planner prompting or atomic-query generation.
- Segment generation with MinerU, DeepSeek OCR, SAM, or another parser.
- Hard-negative mining.
- LoRA training.
- End-to-end MMLongBench evaluation.
- The human evidence-region annotation interface.
- Deleting or rewriting upstream annotations.
- Repackaging source datasets into the Git repository.

The architecture must retain the fields those later phases need, but it must not silently perform those phases.

---

## 3. Approved design decisions

The following decisions are binding unless the user explicitly revises them:

1. Build a self-contained internal corpus first. Preserve original documents/images and deterministic normalized pages.
2. A later public distribution may contain annotations, hashes, source identifiers, provenance, and reconstruction tools rather than every original binary.
3. Keep all source records. Filtering changes manifest membership; it never deletes raw records.
4. V1 focuses on evidence generation that is already constrained by native region anchors or native evidence pages.
5. Questions requiring page discovery, exhaustive document verification, document-global layout reasoning, or unsupported page inference live in separate sets.
6. Original-question difficulty is classified before evidence generation. Global questions do not enter V1 merely because a later planner might decompose them.
7. DUDE boxes are answer-location anchors, not complete evidence.
8. SlideVQA structural boxes are document elements, not QA-specific evidence boxes.
9. TAT-DQA block mappings are supporting fact/operand anchors and may omit interpretive context.
10. MMLongBench is never used for training, prompt selection, hyperparameter selection, sampling-weight selection, or threshold selection.
11. Preserve a permanent document-disjoint calibration set instead of consuming every validation record in the default final fit.
12. Treat projected dataset-size figures as hypotheses until the planner and localization pipeline produce measured counts.

---

## 4. Verified source inventory

All counts in this section were measured from the released annotation artifacts inspected during design. The implementation must download frozen snapshots, recompute the counts, and stop if they differ.

### 4.1 DUDE

Primary references:

- Paper: `https://openaccess.thecvf.com/content/ICCV2023/html/Van_Landeghem_Document_Understanding_Dataset_and_Evaluation_DUDE_ICCV_2023_paper.html`
- Hugging Face release: `https://huggingface.co/datasets/jordyvl/DUDE_loader`
- Annotation archive currently referenced by the official loader: `https://zenodo.org/record/7763635/`

Observed public annotation counts:

| Split | Documents | Questions |
|---|---:|---:|
| train | 2,974 | 23,736 |
| validation | 744 | 6,318 |
| test | 1,299 | 11,402 |

Observed train answer types:

| Type | Count |
|---|---:|
| extractive | 10,679 |
| list/extractive | 1,181 |
| abstractive | 9,092 |
| list/abstractive | 443 |
| not-answerable | 2,341 |

Derived train pools:

```text
native boxed questions     = 10,679 + 1,181 = 11,860
answerable without boxes   = 9,092 + 443    = 9,535
native unanswerable        = 2,341
```

The 11,860 boxed train questions contain 15,293 native boxes. Of those questions, 11,624 refer to one page and 236 refer to multiple pages.

DUDE validation contains 2,917 boxed questions before MMLongBench exclusion. After excluding the 23 DUDE validation documents reused by the frozen MMLongBench source collection, the clean validation pool contains 6,116 total questions and 2,816 boxed questions over 721 documents.

The public test annotations do not provide the labels needed for the proposed training/validation contract. Archive test assets and raw rows, but do not manufacture missing labels.

### 4.2 SlideVQA

Primary references:

- Repository: `https://github.com/nttmdlab-nlp/SlideVQA`
- Hugging Face release: `https://huggingface.co/datasets/NTT-hil-insight/SlideVQA`
- Paper: `https://ojs.aaai.org/index.php/AAAI/article/view/26598`

Observed released counts:

| Split | Decks | Questions |
|---|---:|---:|
| train | 1,919 | 10,617 |
| development | 300 | 1,652 |
| test | 400 | 2,215 |

Other assertions:

```text
decks                         = 2,619
pages                         = 52,380
questions                     = 14,484
structural boxes              = 890,945
train one-page evidence       = 9,381
train two-page evidence       = 1,221
train three-page evidence     = 15
```

The paper's 52,480-page value conflicts with both the release and `2,619 * 20 = 52,380`. Treat 52,380 as the release assertion and record the paper discrepancy in the source audit.

Structural box classes observed in the release include Caption, Diagram, Figure, Image, Obj-text, Other-text, Page-text, Table, and Title. Preserve raw class strings and map them to a canonical class only through a versioned mapping table.

SlideVQA test is held out. The frozen MMLongBench source collection reuses 27 SlideVQA test decks. No SlideVQA train/development deck-ID overlap was found during design.

### 4.3 TAT-DQA

Primary references:

- Project page: `https://nextplusplus.github.io/TAT-DQA/`
- Hugging Face release: `https://huggingface.co/datasets/next-tat/TAT-DQA`
- Repository: `https://github.com/NExTplusplus/TAT-DQA`
- Paper: `https://arxiv.org/abs/2207.11871`

Observed released counts:

| Split | Documents | Questions | Nonempty block mappings |
|---|---:|---:|---:|
| train | 2,207 | 13,251 | 12,876 |
| development | 274 | 1,644 | 1,604 |
| test-gold | 277 | 1,663 | 1,640 |

Observed train answer types:

| Type | Count |
|---|---:|
| span | 5,737 |
| multi-span | 1,656 |
| arithmetic | 5,553 |
| count | 305 |

Preserve each question's `facts`, `derivation`, `scale`, `req_comparison`, and `block_mapping`. Do not infer a missing block mapping from answer-string occurrence during gathering.

Filename/source-name inspection found no direct TAT-DQA/MMLongBench match, but asset-level overlap was not proven absent. The implementation must label this status `identifier_overlap_none_asset_overlap_unverified` until PDF/page/OCR deduplication has run.

### 4.4 MMLongBench-Doc

Primary references:

- Hugging Face release: `https://huggingface.co/datasets/yubo2333/MMLongBench-Doc`
- Repository: `https://github.com/mayubo2333/MMLongBench-Doc`
- Paper: `https://arxiv.org/abs/2407.01523`

Freeze this evaluation snapshot:

```text
repository revision:
2ff6aa9237fc777b6627dc57a486e9225ac5fb86

data/train-00000-of-00001.parquet SHA-256:
bcdac3c96669634c34184814cede4fe57cf7ac0f98dde0e85936394f6a56a02d
```

Observed frozen counts:

| Field | Count |
|---|---:|
| documents | 135 |
| questions | 1,091 |
| one evidence page | 485 |
| multiple evidence pages | 360 |
| empty evidence-page list | 246 |
| `answer_format == None` | 244 |
| unanswerable with nonempty pages | 7 |
| answerable with empty pages | 9 |

Write the 16 page/answerability mismatches to `audits/mmlongbench_exceptions.jsonl`. Do not rewrite the official rows.

The earlier GitHub JSON with 1,082 rows is a different artifact and must not be mixed with this snapshot.

---

## 5. Dataset-set architecture

Every canonical question belongs to exactly one primary generation set for a given classification version. It may also belong to split and audit manifests. Set membership is versioned and reversible.

### 5.1 `v1_direct_region_seeded`

Purpose: easiest automatic evidence-generation inputs because a source-native spatial anchor already exists.

Initial candidates:

```text
DUDE boxed train questions             11,860
TAT-DQA mapped train questions         12,876
pre-filter total                       24,736
```

Eligibility:

- answerable;
- native boxes or fact blocks resolve to existing pages;
- expected support is finite rather than exhaustive;
- no known MMLongBench overlap;
- all geometry passes validation.

Native region anchors are seeds. They do not assert evidence completeness.

### 5.2 `v1_page_conditioned`

Purpose: straightforward page-conditioned generation where the supporting page set is native but QA-specific boxes must be selected.

Initial candidates:

```text
SlideVQA train questions               10,617
```

Eligibility:

- nonempty native `evidence_pages`;
- answerable;
- finite support is expected on those pages;
- no exhaustive document-wide proof is required;
- all evidence pages and structural candidates resolve.

### 5.3 Combined V1 candidate count

```text
24,736 direct-region candidates
+10,617 page-conditioned candidates
=35,353 native-anchor candidates before filtering
```

This number must never be reported as the number of final gold positives. Final V1 positives are the rows that survive asset validation, generation-set classification, later evidence generation, and later sufficiency verification.

### 5.4 `v2_page_grounded_complex`

Purpose: questions with known pages/anchors whose support is more difficult to generate reliably.

Examples:

- multiple disjoint evidence regions;
- support spread over several supplied pages;
- answer-location page that lacks a prerequisite;
- table operands whose headers/scale are not included in native mappings;
- lists and comparisons over several finite regions;
- conditional or bridge questions whose complete native page set is uncertain.

Membership is assigned when a native-anchor candidate fails the V1 simplicity/sufficiency screen but remains page-grounded.

### 5.5 `v3_page_discovery`

Purpose: answerable questions without reliable native evidence pages.

Initial known candidates:

```text
DUDE answerable train without boxes     9,535
TAT-DQA train without block mapping        375
pre-filter total                         9,910
```

These records require full-document page discovery before localization and therefore cannot enter V1.

### 5.6 Global-scope sets

Global questions are separated by the kind of generation/verification machinery they would require.

#### `global_counting`

Examples: number of pages with signatures, total logos, document-wide figures, pages containing tables.

#### `global_presence_absence`

Examples: whether any checkbox exists, whether a signature is present anywhere, whether a requested entity is absent.

#### `global_layout_style`

Examples: page-margin uniformity, orientation consistency, document-wide font or color style.

#### `global_comparison`

Examples: page with the largest table, item appearing most frequently, page with the most signatures.

#### `document_metadata`

Examples: total page count, whole-document organization, metadata-like publication questions without a finite evidence unit.

These questions do not enter V1. They remain available for purpose-built later generation pipelines.

### 5.7 `negative_only`

Initial candidates:

```text
DUDE native unanswerable train questions  2,341
```

Future downstream sources include evidence-page dropout, ColPali misses, topically related insufficient pages, and incomplete-hop candidates. Gathering only stores native unanswerables; it does not manufacture new negatives.

### 5.8 `needs_review` and `invalid`

`needs_review` means the source row and assets are valid but automatic set assignment is not trustworthy. `invalid` means a required invariant is broken, such as a missing document, unresolved page, duplicate source question ID, malformed geometry, or unreadable asset.

Every excluded row must carry a machine-readable reason code.

---

## 6. Canonical identity and schemas

Write canonical tables as Parquet with PyArrow schemas declared in code. JSON examples below are illustrative; implementation must not infer types from the first row.

### 6.1 Stable identifiers

Use full SHA-256 hex values; do not truncate them in persisted tables.

```python
document_uid = sha256(source_dataset + "\0" + source_document_id)
page_uid = sha256(document_uid + "\0" + str(page_index_zero_based))
question_uid = sha256(source_dataset + "\0" + source_question_id)
anchor_uid = sha256(question_uid + "\0" + anchor_type + "\0" + source_anchor_id)
```

The content hash of a PDF/image is separate from `document_uid`. Two source records may refer to identical binary content while retaining separate source identities.

### 6.2 `source_snapshots.parquet`

Required columns:

```text
source_dataset: string
source_component: string
source_url: string
source_revision: string
downloaded_at_utc: timestamp[us, UTC]
local_relpath: string
sha256: string
byte_count: int64
media_type: string
license_identifier: string
license_text_relpath: string|null
redistribution_status: string
retrieval_method: string
```

`source_revision` must be a repository commit, immutable artifact version, or explicit archive checksum. The literal string `main` is forbidden in a frozen run.

### 6.3 `documents.parquet`

Required columns:

```text
document_uid: string
source_dataset: string
source_document_id: string
source_split: string
canonical_split: string
original_asset_relpath: string
original_asset_sha256: string
original_media_type: string
page_count: int32
source_url: string|null
source_title: string|null
source_metadata_json: string
exact_duplicate_group: string|null
near_duplicate_group: string|null
mmlongbench_overlap_status: string
asset_status: string
```

Allowed `asset_status` values:

```text
valid
missing
unreadable
page_count_mismatch
hash_mismatch
quarantined
```

### 6.4 `pages.parquet`

Required columns:

```text
page_uid: string
document_uid: string
page_index: int32
source_page_number: int32|null
original_page_relpath: string|null
normalized_page_relpath: string
normalized_page_sha256: string
source_width: int32
source_height: int32
normalized_width: int32
normalized_height: int32
render_dpi: int32|null
scale_x: float64
scale_y: float64
page_phash: string
ocr_text_sha256: string|null
render_status: string
```

Canonical `page_index` is zero-based. Preserve every source's original page-number convention in `source_page_number` and source metadata.

### 6.5 `questions.parquet`

Required columns:

```text
question_uid: string
document_uid: string
source_dataset: string
source_question_id: string
source_split: string
question: string
answers_json: string
answer_type: string
answerable: bool|null
native_evidence_pages: list<int32>
native_anchor_type: string
support_completeness: string
generation_set: string
generation_set_version: string
classification_confidence: float32|null
classification_reason: string
requires_exhaustive_verification: bool
finite_positive_regions_expected: bool|null
eligible_v1_candidate: bool
exclusion_reason_codes: list<string>
source_payload_json: string
```

Allowed `native_anchor_type` values:

```text
dude_answer_location
slidevqa_evidence_slide
tat_fact_block_mapping
none
```

Allowed `support_completeness` values:

```text
unknown
sufficient_verified
incomplete
requires_additional_pages
not_applicable
```

`support_completeness` remains `unknown` during gathering unless a source explicitly supplies complete support. Gathering must not promote it to `sufficient_verified`.

### 6.6 `native_anchors.parquet`

Required columns:

```text
anchor_uid: string
question_uid: string
document_uid: string
page_uid: string
anchor_type: string
source_anchor_id: string
source_bbox_json: string|null
bbox_norm_1000: fixed_size_list<int32>[4]|null
source_coordinate_system: string|null
source_text: string|null
source_role: string
source_payload_json: string
geometry_status: string
```

Use `[x0, y0, x1, y1]` for `bbox_norm_1000`, clamped to `[0,1000]`, with strict `x1 > x0` and `y1 > y0`. Preserve original geometry verbatim in `source_bbox_json`.

### 6.7 `manifest_membership.parquet`

Required columns:

```text
question_uid: string
manifest_name: string
manifest_version: string
included: bool
reason_codes: list<string>
assigned_at_utc: timestamp[us, UTC]
```

Manifest rows are append-only across versions. A newer classification does not erase the previous decision.

### 6.8 `audit_events.jsonl`

Every fail-closed or quarantine decision writes:

```json
{
  "event_type": "invalid_anchor_geometry",
  "severity": "error",
  "source_dataset": "DUDE",
  "document_uid": "...",
  "question_uid": "...",
  "details": {"reason": "x1 <= x0"},
  "created_at_utc": "..."
}
```

---

## 7. Internal artifact layout

Large artifacts must live outside Git, normally under `/scratch/$USER`. The run root is:

```text
/scratch/$USER/segment_reranker_corpus/<run_id>/
```

Required layout:

```text
<run_root>/
  sources/
    dude/
    slidevqa/
    tat_dqa/
    mmlongbench/
  assets/
    originals/
      <source_dataset>/<document_uid>/...
    normalized_pages/
      <source_dataset>/<document_uid>/<page_index:05d>.png
  tables/
    source_snapshots.parquet
    documents.parquet
    pages.parquet
    questions.parquet
    native_anchors.parquet
    manifest_membership.parquet
  manifests/
    v1_direct_region_seeded.parquet
    v1_page_conditioned.parquet
    v2_page_grounded_complex.parquet
    v3_page_discovery.parquet
    global_counting.parquet
    global_presence_absence.parquet
    global_layout_style.parquet
    global_comparison.parquet
    document_metadata.parquet
    negative_only.parquet
    needs_review.parquet
    train_development.parquet
    validation_development.parquet
    permanent_calibration.parquet
    train_final_default.parquet
    train_final_maximal.parquet
    mmlongbench_evaluation.parquet
  audits/
    source_counts.json
    source_schema.json
    asset_integrity.json
    render_audit.json
    geometry_audit.json
    classification_audit.json
    split_audit.json
    leakage_report.json
    mmlongbench_exceptions.jsonl
    audit_events.jsonl
  provenance/
    source_revisions.json
    source_licenses/
    download_manifest.json
    environment.txt
  run_manifest.json
```

`run_manifest.json` records the Git commit, command, configuration hash, source snapshot hashes, counts, Python version, dependency versions, start/end UTC, host, and every output-file SHA-256.

---

## 8. Deterministic rendering and coordinates

### 8.1 Original assets

Copy downloaded original PDFs/images without modification. Verify their SHA-256 before and after staging. Never treat a normalized page as the original asset.

### 8.2 PDF pages

Use the repository's established Poppler `pdftoppm` path unless an implementation experiment proves it cannot render a frozen fixture. The implementation must pin and record `pdftoppm -v` output.

Canonical render policy:

```text
render at 200 DPI
convert to RGB
if long edge > 2400 px, resize once to long edge 2400 with PIL LANCZOS
never upscale
save PNG with optimize=False and compress_level=6
```

### 8.3 Slide images

Decode the source image, apply EXIF orientation, convert to RGB, preserve aspect ratio, cap the long edge at 2400 px without upscaling, and save using the same PNG parameters.

### 8.4 Coordinate normalization

For source dimensions `(W, H)` and source box `[x0,y0,x1,y1]`:

```python
nx0 = round(x0 / W * 1000)
ny0 = round(y0 / H * 1000)
nx1 = round(x1 / W * 1000)
ny1 = round(y1 / H * 1000)
```

Use Python's round-half-even behavior, clamp to `[0,1000]`, and reject boxes with nonpositive area. If a source uses `[left, top, width, height]`, convert to corners before normalization and record that convention.

Downstream consumers use only normalized geometry and the normalized page image. Original geometry remains available for audit.

---

## 9. Source adapters

Each source adapter converts a frozen source snapshot to canonical rows. It must not download files or write global state during row conversion.

### 9.1 Proposed package structure

```text
scripts/segment_dataset/
  __init__.py
  schemas.py
  ids.py
  snapshots.py
  assets.py
  render.py
  geometry.py
  dedup.py
  classification.py
  manifests.py
  audit.py
  sources/
    __init__.py
    dude.py
    slidevqa.py
    tat_dqa.py
    mmlongbench.py
scripts/build_segment_reranker_corpus.py
```

Tests mirror these responsibilities:

```text
tests/segment_dataset/
  test_schemas.py
  test_ids.py
  test_snapshots.py
  test_render.py
  test_geometry.py
  test_dedup.py
  test_classification.py
  test_manifests.py
  test_audit.py
  test_dude_adapter.py
  test_slidevqa_adapter.py
  test_tat_dqa_adapter.py
  test_mmlongbench_adapter.py
  test_end_to_end_fixture.py
```

### 9.2 Required interfaces

```python
@dataclass(frozen=True)
class SourceBundle:
    source_name: str
    snapshot_root: Path
    revision: str
    files: tuple[SourceFile, ...]

class SourceAdapter(Protocol):
    def validate_snapshot(self, bundle: SourceBundle) -> SourceAudit: ...
    def iter_documents(self, bundle: SourceBundle) -> Iterable[DocumentRecord]: ...
    def iter_questions(self, bundle: SourceBundle) -> Iterable[QuestionRecord]: ...
    def iter_native_anchors(self, bundle: SourceBundle) -> Iterable[NativeAnchorRecord]: ...
```

The adapter must emit rows in deterministic source-ID order.

### 9.3 DUDE adapter rules

- `docId` is the source document ID.
- `questionId` is the source question ID.
- Preserve `answers`, `answers_variants`, `answer_type`, and the raw annotation row.
- Convert each answer box from `[left, top, width, height]` semantics only after confirming the actual loader schema.
- Native evidence pages are the unique sorted page values from nonempty boxes.
- Boxed answerable rows initially route to `v1_direct_region_seeded` unless a global-scope classifier overrides them.
- Answerable rows without boxes route to `v3_page_discovery` unless another exclusion applies.
- `not-answerable` rows route to `negative_only`.
- Test rows without public labels remain archived and ineligible.

### 9.4 SlideVQA adapter rules

- `deck_name` is the source document ID.
- `qa_id` is the source question ID.
- Preserve `deck_url`, `image_urls`, answer, arithmetic expression, and raw row.
- Evidence pages are source one-based values. Convert to zero-based canonical page indices only after checking every value is in `[1,20]` for the source release.
- Resolve every structural box page and normalize its geometry.
- QA rows with finite support and valid evidence pages initially route to `v1_page_conditioned`.
- Do not treat structural boxes as positive QA evidence.
- Entire test split is ineligible for training.

### 9.5 TAT-DQA adapter rules

- `doc.uid` is the source document ID.
- question `uid` is the source question ID.
- Preserve `doc.source`, `doc.page`, facts, derivation, scale, answer type, comparison flag, and raw mapping.
- Resolve every block UUID against the converted document JSON and identify its page.
- A row with a nonempty mapping is V1-direct only if every mapping resolves and geometry is valid.
- A row with an empty/unresolved mapping routes to `v3_page_discovery` or `invalid`, depending on whether source corruption or simply absent supervision caused the failure.
- Preserve table/text block boundaries. Do not merge blocks during gathering.

### 9.6 MMLongBench adapter rules

- Use only the frozen 1,091-row revision.
- Parse stringified evidence-page/source arrays with `ast.literal_eval`; never use `eval`.
- Classify answerability using `answer_format == "None"`, not empty evidence pages.
- Preserve official rows exactly in the source payload.
- Emit the 16 mismatches to the exception audit.
- Mark every MMLongBench document and question `canonical_split=evaluation` and ineligible for every training manifest.

---

## 10. Generation-set classification

Classification occurs on the original question before evidence generation. It is source-aware, versioned, and auditable.

### 10.1 Deterministic first pass

Metadata determines the initial candidate class:

```text
DUDE boxed answerable       -> v1_direct_region_seeded candidate
DUDE answerable unboxed     -> v3_page_discovery
DUDE unanswerable           -> negative_only
SlideVQA valid pages        -> v1_page_conditioned candidate
TAT mapped                  -> v1_direct_region_seeded candidate
TAT unmapped                -> v3_page_discovery
```

### 10.2 Scope classifier

The classifier must distinguish finite evidence from exhaustive/global verification. High-precision rules may nominate but never permanently discard rows.

Required output schema:

```json
{
  "question_uid": "...",
  "classification_version": "scope_v1",
  "generation_set": "global_counting",
  "finite_positive_regions_expected": true,
  "requires_exhaustive_verification": true,
  "native_page_labels_sufficient": false,
  "confidence": 0.97,
  "reason": "The answer requires checking every page for a signature."
}
```

### 10.3 High-precision signals

Signals include, but are not limited to:

- “all pages”, “every page”, “throughout the document”;
- “how many pages”;
- negative existence or absence;
- uniform margins, orientation, fonts, or color style;
- largest/most/fewest across pages;
- global page count or document organization.

Do not classify by a single keyword. “Margin of error” is not a page-margin question. Rules must include phrase/context constraints and write their matched rule IDs.

### 10.4 Human audit

Select 750 questions deterministically using seed `20260715`:

```text
250 DUDE
250 SlideVQA
250 TAT-DQA
```

Within each source, stratify across initial generation set, answer type, native page count, and rule/model disagreements. Oversample global/count/list/multi-page/layout cases. Sampling is document-aware: no single document may contribute more than five audit questions unless the source contains too few documents for the requested stratum.

Report per source and per class:

- precision;
- recall against human labels;
- macro F1;
- disagreement count;
- V1 false-admission rate;
- V1 false-rejection rate.

The V1 classifier is acceptable only if human-estimated false admission into V1 is at most 5% with a 95% Wilson upper bound at most 8%. Otherwise route uncertain rows to `needs_review` and revise the classifier before evidence generation.

---

## 11. Split architecture

Splits are document-disjoint. A document and every question derived from it must share one canonical split.

### 11.1 Development manifests

Pre-filter candidate counts:

```text
train_development
  DUDE train boxed          11,860
  SlideVQA train            10,617
  TAT-DQA train mapped      12,876
  total                     35,353

validation_development
  clean DUDE val boxed       2,816
  SlideVQA dev               1,652
  TAT-DQA dev mapped         1,604
  total                      6,072
```

Actual manifest counts are recomputed after classification and validation. Audits must show both pre-filter and post-filter counts.

### 11.2 Permanent calibration

From eligible development-validation documents, select approximately 400 questions per source using whole documents:

1. Compute `sha256("20260715" + document_uid)`.
2. Sort documents by that hash.
3. Add complete documents until the source reaches or first exceeds 400 eligible questions.
4. Never split a document to hit the target exactly.

Expected calibration size is approximately 1,200–1,500 because whole-document selection may overshoot.

### 11.3 Final manifests

`train_final_default` contains:

- every eligible `train_development` question;
- eligible validation questions not assigned to permanent calibration.

Expected pre-filter size is approximately 39,900–40,200.

`train_final_maximal` contains all eligible train and validation questions and is provided for controlled experiments. Its pre-filter upper bound is 41,425. It is not the default reported fit.

### 11.4 MMLongBench

All 1,091 frozen questions belong exclusively to `mmlongbench_evaluation`. No training or validation manifest may reference their document content, exact source documents, or known duplicate pages.

---

## 12. Leakage and deduplication

### 12.1 Known source reuse

The implementation must reproduce these identifier-level findings:

```text
MMLongBench vs DUDE train             0 matching documents
MMLongBench vs DUDE validation       23 matching documents
MMLongBench vs SlideVQA train         0 matching decks
MMLongBench vs SlideVQA development   0 matching decks
MMLongBench vs SlideVQA test          27 matching decks
```

Store the exact IDs in a generated denylist; do not rely on counts alone.

### 12.2 Exact document hashes

Group identical original asset SHA-256 values. If a training candidate and evaluation document share a hash, quarantine the training document and fail the final leakage gate.

### 12.3 Page similarity

For every normalized page compute:

- SHA-256;
- 64-bit perceptual hash;
- normalized OCR text fingerprint when OCR exists.

Flag:

```text
exact page SHA match
perceptual Hamming distance <= 4
OCR MinHash Jaccard estimate >= 0.90
```

Threshold hits are candidates for audit, not automatic duplicate declarations, except exact SHA matches.

### 12.4 Metadata families

Normalize company/report title/year where available. Write potential same-company/same-year and same-template families. Do not exclude a family automatically; report family-aware evaluation and allow an explicit strict-family manifest later.

### 12.5 Final leakage gate

Before freezing any train manifest, assert:

- no exact evaluation document hash in training;
- no exact evaluation page hash in training;
- all known DUDE/SlideVQA source IDs denied;
- every unresolved near-duplicate has an audit disposition;
- TAT-DQA/MMLongBench asset-overlap status is no longer `unverified`.

---

## 13. Storage and resource planning

Observed current repository sizes at design time:

| Source | Size |
|---|---:|
| DUDE | 21.4 GB |
| SlideVQA | 10.9 GB |
| TAT-DQA | 1.5 GB |
| MMLongBench | 0.66 GB |
| total | about 34.5 GB |

The implementation must obtain current sizes from frozen repository manifests and write them to the source audit.

Capacity requirement:

```text
minimum practical workspace       150 GB
recommended workspace             200-250 GB
```

This includes frozen downloads, extracted originals, normalized PNG pages, intermediate staging, and audit artifacts. Future segment crops require separate capacity planning.

---

## 14. Downstream interface

The gathering corpus does not directly satisfy the current classifier contract in `sol/task_spec/current_design_architecture_plan.md`. A later localization/segmentation adapter will produce:

```text
labels/semantic_sections.jsonl
labels/queries.jsonl
labels/query_section_labels.jsonl
splits/{train,validation,test}.jsonl
splits/page_assignments.jsonl
audits/labeling_audit.json
audits/split_audit.json
```

Required lineage carried into that adapter:

```text
question_uid
document_uid
page_uid
source_dataset
source_question_id
native_anchor_type
support_completeness
generation_set
source snapshot hashes
normalized page SHA-256
```

The downstream model input must not receive ground-truth answers, native gold geometry, source completeness labels, or audit scores unless a separately approved experiment explicitly uses them. They are annotation-generation and evaluation metadata.

---

## 15. MMLongBench evaluation dependency

The full frozen 1,091-question benchmark measures end-to-end transfer. It does not contain region evidence and therefore cannot directly measure segment Recall@k.

A later, separately approved human annotation phase must create approximately 400 answerable region-grounded examples stratified by:

- text, table, chart, figure/image, and layout;
- one-page and multi-page support;
- DUDE-derived, SlideVQA-derived, finance, academic, and newly collected document origins;
- answer type and document length.

Required human labels:

- complete supporting regions;
- answer-bearing versus contextual regions;
- table headers, labels, captions, and scales;
- evidence roles;
- alternative valid evidence sets;
- page/region sufficiency.

Automatically generated training labels must never be treated as gold evaluation labels for this subset.

---

## 16. Implementation sequence

Each task below produces a reviewable, independently testable deliverable. Implementation should use test-driven development and focused pytest targets.

### Task 1: Freeze schemas and identifiers

Files:

```text
scripts/segment_dataset/schemas.py
scripts/segment_dataset/ids.py
tests/segment_dataset/test_schemas.py
tests/segment_dataset/test_ids.py
```

Deliverables:

- explicit PyArrow schemas for all canonical tables;
- deterministic SHA-256 identity functions;
- enum validation;
- round-trip Parquet tests;
- duplicate-ID failure tests.

Acceptance command:

```bash
python -m pytest -q tests/segment_dataset/test_schemas.py tests/segment_dataset/test_ids.py --tb=short
```

### Task 2: Snapshot and acquisition layer

Files:

```text
scripts/segment_dataset/snapshots.py
scripts/build_segment_reranker_corpus.py
tests/segment_dataset/test_snapshots.py
```

Deliverables:

- immutable source revision resolution;
- download manifests;
- SHA-256 verification;
- resumable staging without silently accepting partial files;
- `--offline` mode that uses only verified local snapshots;
- rejection of literal `main` in a frozen run.

Unit tests must mock network access. A separate opt-in integration command verifies live source reachability.

### Task 3: Source adapters

Files:

```text
scripts/segment_dataset/sources/dude.py
scripts/segment_dataset/sources/slidevqa.py
scripts/segment_dataset/sources/tat_dqa.py
scripts/segment_dataset/sources/mmlongbench.py
tests/segment_dataset/test_dude_adapter.py
tests/segment_dataset/test_slidevqa_adapter.py
tests/segment_dataset/test_tat_dqa_adapter.py
tests/segment_dataset/test_mmlongbench_adapter.py
```

Deliverables:

- deterministic canonical rows from minimal synthetic source fixtures;
- expected-count assertions against frozen real snapshots;
- raw source payload preservation;
- source page-number conversion tests;
- the 16 MMLongBench exception rows.

### Task 4: Asset materialization and rendering

Files:

```text
scripts/segment_dataset/assets.py
scripts/segment_dataset/render.py
scripts/segment_dataset/geometry.py
tests/segment_dataset/test_render.py
tests/segment_dataset/test_geometry.py
```

Deliverables:

- original asset staging;
- deterministic 200-DPI/2400-pixel PNG output;
- page-count verification;
- coordinate normalization;
- source-to-normalized transform metadata;
- corrupted PDF/image and invalid box tests.

### Task 5: Deduplication and evaluation denylist

Files:

```text
scripts/segment_dataset/dedup.py
tests/segment_dataset/test_dedup.py
```

Deliverables:

- exact PDF/page hashing;
- perceptual page hashing;
- OCR fingerprint interface;
- identifier denylist generation;
- MMLongBench collision reports;
- fail-closed final leakage gate.

### Task 6: Generation-set classifier

Files:

```text
scripts/segment_dataset/classification.py
tests/segment_dataset/test_classification.py
```

Deliverables:

- metadata-first routing;
- contextual high-precision global rules;
- structured model-classifier import/export interface;
- stable reason codes;
- deterministic 750-question audit selection;
- no keyword-only destructive filtering.

### Task 7: Splits and manifests

Files:

```text
scripts/segment_dataset/manifests.py
tests/segment_dataset/test_manifests.py
```

Deliverables:

- every approved generation-set manifest;
- development, validation, permanent-calibration, and final manifests;
- whole-document split assertions;
- deterministic calibration selection;
- manifest version lineage.

### Task 8: Audit and end-to-end fixture

Files:

```text
scripts/segment_dataset/audit.py
tests/segment_dataset/test_audit.py
tests/segment_dataset/test_end_to_end_fixture.py
```

Deliverables:

- count, schema, asset, render, geometry, classification, split, and leakage audits;
- one tiny multi-source fixture exercising every primary generation set;
- offline rerun equality by file hash;
- a final run manifest containing hashes of all outputs.

### Task 9: SOL execution wrappers

Before implementing this task, read `docs/SOL_INSTRUCTIONS.md`, `sol/AGENTS.md`, `agent-context/SOL.md`, and `sol/CURRENT_SOL_TASK.md`.

Files:

```text
sol/build_segment_reranker_corpus.sbatch
sol/audit_segment_reranker_corpus.sbatch
tests/test_segment_reranker_corpus_sol.py
```

Deliverables:

- source acquisition separated from CPU-heavy rendering/auditing where appropriate;
- `/scratch/$USER` outputs and caches;
- syntax tests and required environment assertions;
- no automatic job submission from unit tests.

---

## 17. Failure policy

Fail closed for:

- source revision drift;
- source count mismatch;
- schema key/type drift;
- missing assets;
- unreadable assets;
- page-count disagreement;
- unresolved page references;
- duplicate canonical IDs;
- invalid geometry;
- exact MMLongBench collision;
- unreviewed near-duplicate at final freeze;
- cross-split document leakage;
- nonreproducible manifest generation.

Quarantine rather than abort the entire run only when the error is row/document-local and the source audit explicitly counts it. Source-wide schema/count drift aborts the run.

Never add a fallback that silently infers a box, page, answerability label, or source identifier.

---

## 18. Acceptance criteria

The corpus is implementation-complete when all of the following hold:

1. Frozen source snapshots and hashes are present.
2. Observed source counts either match Section 4 or an explicitly approved revision update is documented.
3. Every question references exactly one canonical document.
4. Every native evidence page resolves to a canonical page.
5. Every persisted native geometry is valid and reversible to source coordinates within rounding tolerance.
6. Every question has a primary generation set and reason.
7. Raw records are preserved for every canonical question.
8. No known MMLongBench document/page appears in training.
9. TAT-DQA/MMLongBench asset overlap is resolved beyond filename matching.
10. Development, validation, calibration, and final manifests are document-disjoint where required.
11. The MMLongBench snapshot exactly matches the pinned 1,091-row hash.
12. A second offline run over the same snapshots produces identical canonical-table and manifest hashes.
13. The 750-question classification audit meets the V1 false-admission criterion or uncertain rows are excluded from V1.
14. All focused tests and the end-to-end fixture pass.
15. `agent-context/INDEX.md` and this module remain consistent with the implemented paths.

---

## 19. Research checklist for the implementing agent

Before implementation, independently verify and record:

- current official source URLs and immutable revisions;
- the exact current annotation schemas rather than relying on paper tables;
- source download sizes and available disk space;
- whether source PDFs/images are contained directly or reconstructed from URLs;
- page-number conventions for every source;
- coordinate conventions for every box type;
- availability and versions of Poppler, Pillow, PyArrow, Hugging Face Hub, and OCR fingerprint dependencies;
- exact MMLongBench leaderboard/evaluation revision;
- source document IDs reused by MMLongBench;
- actual asset-level TAT-DQA/MMLongBench overlap;
- whether the repository's current Python environment supports all proposed dependencies without a production dependency addition.

If a dependency addition or material architecture change is required, stop and request approval before modifying production environments.

---

## 20. Reporting language

Use these terms consistently:

- “native-anchor candidate” before generated support has been verified;
- “answer-location anchor” for DUDE boxes;
- “evidence-slide label” for SlideVQA pages;
- “fact-block anchor” for TAT-DQA mappings;
- “verified positive” only after the later evidence-generation and sufficiency pipeline passes;
- “global-scope” for questions requiring exhaustive or document-wide reasoning;
- “MMLongBench transfer evaluation” for the full benchmark;
- “segment-level evaluation” only for a human region-grounded subset.

Do not report generated pair counts as independent human questions. Always report unique documents, pages, original questions, atomic queries, positives, negatives, and optimizer exposures separately.
