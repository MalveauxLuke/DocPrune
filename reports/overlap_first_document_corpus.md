# Overlap-First Document Corpus: Metadata Ledger

## Scope and available data

The ledger compares only these locally available inputs:

| Input | Accepted question observations | Parent documents |
|---|---:|---:|
| Visual-CoT `docvqa_cot_train.jsonl` | 33,453 | 5,131 |
| Visual-CoT `dude_cot_train.jsonl` | 11,735 | 2,904 |
| Visual-CoT `infographicsvqa_cot_train.jsonl` | 15,055 | 3,805 |
| BoundingDocs `DUDE` | 4,512 | 2,582 |
| BoundingDocs `MP-DocVQA` | 31,597 | 5,203 |
| BoundingDocs `SP-DocVQA` | 419 | 248 parent documents from 266 page records |

All 60,243 Visual-CoT manifest rows and all 8,051 selected BoundingDocs
document rows were read. Their 96,771 question observations were accepted and
preserved, with no exclusion. The pinned BoundingDocs Parquets physically contain 2,582 DUDE
documents, not the stale 2,583 value in the release card.

Visual-CoT metadata came from
`/scratch/lmalveau/evidence_dino_units/data/raw/visual_cot_repo/metadata`.
BoundingDocs came from pinned commit
`dd2c05ae4d516bff617409ceb942d49f22452852` under
`/scratch/lmalveau/boundingdocs_document_qa/raw`.

## Linking and canonicalization

The script uses exact metadata identities before any image or text similarity:

- Visual-CoT DocVQA and DUDE remove the final numeric page suffix from the
  image stem. DUDE's hash is otherwise unchanged.
- Visual-CoT InfographicVQA uses the full image stem.
- BoundingDocs DUDE and MP-DocVQA use `doc_id` directly.
- BoundingDocs SP-DocVQA removes its final page suffix to recover the parent
  DocVQA document. Its 266 page records represent 248 parents.
- The document families are `docvqa`, `dude`, and `infographicsvqa`.
- Questions are normalized with Unicode NFKC, case folding, and whitespace
  collapse. Punctuation is preserved. A question key always includes the
  document family and parent ID, so identical text on different documents is
  never merged.

Source page conventions remain visible. Visual-CoT DUDE filename pages are
zero-based and are converted to one-based values only in
`canonical_answer_pages`; the original suffix remains in `answer_pages`.
SP-DocVQA's source-local answer page remains `1`, while its parent-relative
page comes from the `doc_id` suffix. Sample inspection confirmed these rules.

Every canonical question retains the full source observations: synthetic
stable observation ID, native question ID when one exists, original question,
answers, possible answers, raw boxes and formats, normalized boxes, source and
canonical answer pages, OCR reference, original split, image reference, and
provenance. Visual-CoT has no native question ID or OCR pointer in these
manifests, so those fields are explicitly null rather than inferred.

## Measured overlap

| Dataset pair | Shared parent documents | Duplicate normalized questions |
|---|---:|---:|
| Visual-CoT DocVQA ↔ BoundingDocs MP-DocVQA | 4,743 | 23,002 |
| Visual-CoT DocVQA ↔ BoundingDocs SP-DocVQA | 216 | 342 |
| Visual-CoT DUDE ↔ BoundingDocs DUDE | 2,058 | 3,594 |

No parent-ID overlap was found between MP-DocVQA and SP-DocVQA after SP page
suffix normalization. InfographicVQA has no counterpart among the three
selected BoundingDocs sources.

The union contains 12,856 canonical documents and 66,551 canonical questions.
There are 26,938 cross-dataset duplicate questions and 39,613 single-dataset
canonical questions.

| Family | Canonical documents | Canonical questions | Cross-dataset duplicate questions | Ecosystem-unmatched documents |
|---|---:|---:|---:|---:|
| DocVQA | 5,623 | 38,846 | 23,344 | 664 |
| DUDE | 3,428 | 12,651 | 3,594 | 1,370 |
| InfographicVQA | 3,805 | 15,054 | 0 | 3,805 |

Unmatched document memberships by dataset are: Visual-CoT DocVQA 172,
Visual-CoT DUDE 846, Visual-CoT InfographicVQA 3,805, BoundingDocs DUDE 524,
BoundingDocs MP-DocVQA 460, and BoundingDocs SP-DocVQA 32.

## Conflicts and review boundary

The review manifest contains 27,312 canonical question groups:

- 3,011 have disjoint normalized answer sets;
- 8,799 have different parent-relative answer-page sets; and
- 27,312 have different exact normalized box sets.

The cross-dataset portions are 3,594 DUDE groups, 23,002 MP-DocVQA groups,
and 342 SP-DocVQA groups. Another 374 conflicts occur among repeated
observations inside one dataset. Exact box disagreement is deliberately broad:
different sources may annotate different occurrences or extents, so it means
"retain for review," not "join is invalid." Nothing selects a winning answer,
page, or box.

Three joins from each linked dataset pair were manually inspected. The parent
IDs and normalized questions matched, original IDs/questions remained present,
and the SP/DUDE page-basis conversions behaved as intended. Three
InfographicVQA unmatched records were also inspected. No ambiguous native-ID
join or schema exclusion was observed.

## Outputs and reproducibility

The reusable builder is `scripts/build_overlap_first_corpus.py`. Full outputs
are under `/home/lmalveau/overlap_first_document_corpus/run1`:

- `canonical_documents.jsonl`
- `canonical_questions.jsonl`
- `overlap_questions.jsonl`
- `unique_questions.jsonl`
- `conflicts_review.jsonl`
- `unmatched_records.jsonl`
- `excluded_records.jsonl`
- `summary.json`

Two complete runs produced identical SHA-256 hashes for every output. The
focused test suite covers identity rules, source adapters, annotation
preservation, exclusions, document-scoped deduplication, conflict emission,
complete accounting, and deterministic output.

## V1 materialization, OCR, and splits

The final manifest package is
`/home/lmalveau/overlap_first_document_corpus/v1`. Visual-CoT remains the base
corpus. The additive BoundingDocs selection contributes 8,085 questions and
6,191 selected page positions: 1,732 pages from BoundingDocs-only documents
and 4,459 pages supporting new questions on shared documents.

V1 preserves all 12,856 canonical documents and all 66,551 canonical
questions. Of those questions, 65,787 map to at least one resolved,
OCR-backed page and are marked `usable_in_v1=true`; 764 observations outside
the frozen resolved Visual-CoT population are retained with
`usable_in_v1=false` and copied to `excluded_questions.jsonl`. This avoids
silently admitting malformed source annotations while preserving the ledger.

The package has 23,043 canonical page positions backed by 23,006 unique image
contents and DeepSeek-OCR-2 records. The difference is intentional: distinct
canonical pages can contain identical bytes and share one content-addressed
OCR result. Every page path exists, every page has a completed OCR record, and
all OCR artifact hashes were recomputed during packaging.

Splits are deterministic and grouped by `canonical_document_id`. DocVQA and
DUDE use an 80/10/10 SHA-256 train/validation/test assignment, preventing
document leakage. InfographicsVQA is side-stepped into a dedicated
`infographicsvqa_holdout` split and never enters the primary three splits.
TextVQA, TextCaps, and SROIE are outside this merged document-QA V1. Existing
conflict annotations and OCR quality flags are preserved without adjudication
or silent retry.

| Split | Documents | Usable questions | Pages |
|---|---:|---:|---:|
| train | 7,272 | 40,963 | 15,584 |
| validation | 907 | 5,023 | 1,911 |
| test | 872 | 4,748 | 1,743 |
| infographicsvqa_holdout | 3,805 | 15,053 | 3,805 |

The deterministic builder is `scripts/build_overlap_first_v1.py`. A random
eight-page visual review is under
`outputs/evidence_dino_ocr_visualizations/v1_new_ocr_review8`; it includes
ordinary, repetition-flagged, truncation-flagged, dense, and rotated pages.
