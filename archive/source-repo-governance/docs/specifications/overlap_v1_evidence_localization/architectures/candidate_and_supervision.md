# Candidate and Supervision Architecture

## Role

This file defines the common immutable data and candidate interface consumed by
every arm in the overlap-first V1 evidence-localization program. Model
comparisons are invalid if they use different candidate revisions, target
semantics, question text, supplied pages, or held-out manifests.

The task is:

```text
known page + original question + frozen semantic candidates
  -> rank or select answer-bearing candidate IDs
```

It is page-conditional localization, not open-corpus retrieval.

## Immutable V1 source

Source package:

```text
/home/lmalveau/overlap_first_document_corpus/v1
```

Verified package counts:

| Split | Documents | Usable questions | Page positions |
|---|---:|---:|---:|
| train | 7,272 | 40,963 | 15,584 |
| validation | 907 | 5,023 | 1,911 |
| internal test | 872 | 4,748 | 1,743 |
| `infographicsvqa_holdout` | 3,805 | 15,053 | 3,805 |

The complete ledger contains 12,856 canonical documents, 66,551 canonical
questions, 65,787 usable questions, 764 quarantined questions, and 23,043
canonical page positions backed by 23,006 unique OCR/image contents. Consumers
must require `usable_in_v1=true` and preserve the deterministic
document-grouped split.

V1 is immutable. Eligibility filtering, exclusions, candidate relations,
alias serialization, and training groups are derived artifacts. No stage may
rewrite, delete, repair, relabel, or silently omit V1 rows.

## Supervision meaning

V1 answer boxes are accepted answer-bearing anchors. They are a high-precision
lower bound on relevant evidence, not audited complete evidence. They may omit
headers, qualifiers, legends, operands, disambiguating context, or other useful
support. Multiple boxes may be:

- members of one answer span;
- alternative occurrences of the same accepted answer;
- annotations from different sources with disagreements; or
- genuinely related locations, without proving that their union is necessary.

The binding relation meanings are:

```text
positive_anchor:
  candidate covers an accepted answer-location alternative

partial_anchor:
  candidate overlaps an accepted answer location but fails the coverage gate

verified_non_anchor:
  audited candidate that covers no accepted answer alternative and passes the
  program's false-negative guards

unverified_context:
  plausible or unresolved candidate that receives no negative loss
```

These labels must never be renamed `complete_evidence` and `irrelevant`.
There is no valid complete-evidence, necessity, sufficiency, no-evidence, or
multi-hop target in V1.

## DeepSeek-OCR-2 source contract

The candidate builder starts from packaged DeepSeek-OCR-2 grounded output. The
inference provenance that produced the source artifacts is:

| Field | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-OCR-2` |
| Revision | `aaa02f3811945a91062062994c5c4a3f4c0af2b0` |
| Prompt | `<image>\n<|grounding|>Convert the document to markdown.` |
| Base size | `1024` |
| Image size | `768` |
| Crop mode | enabled |
| Evaluation mode | enabled |
| Maximum new tokens | `8192` |

The source run records input PNG hash, model/revision, prompt, parameters,
attempt, raw-output hash, runtime, GPU, and Slurm job. Resume is valid only when
provenance and raw hashes match.

The Python candidate implementation, grounded-output parser, focused tests,
and this architecture must remain synchronized. The existing sources of truth
are:

- `scripts/document_parsing/deepseek_runner.py`;
- `scripts/document_parsing/grounding.py`;
- `scripts/document_parsing/semantic_sections.py`;
- `tests/test_semantic_sections.py`; and
- `docs/specifications/deepseek_semantic_segmentation.md`.

## Semantic candidate construction

The current segmentation was designed for structured academic/report pages.
It is not assumed to transfer unchanged to arbitrary webpages, magazines,
forms, receipts, slides, or natural-image scenes.

### Heading-led text

A Markdown block beginning with `#` through `######` starts a section. Heading
level marks a boundary but does not create a trusted nested hierarchy.

For each heading:

1. include the heading;
2. include following non-caption `text` and `formula` blocks;
3. stop before the next Markdown heading; and
4. emit `headed_text`.

Do not split by token count, word count, paragraph count, columns, coordinate
gaps, bounding-box size, or sliding windows. Nonempty material before the first
qualifying heading becomes one `document_preamble`.

### Pages without headings

Do not infer headings from font or geometry. Every nonempty DeepSeek `title`,
`text`, or `formula` block becomes one `deepseek_paragraph`. Do not create
overlapping paragraph windows during preprocessing; conditional adjacency is a
separate future inference strategy.

### Visual runs and captions

Recognized visual objects are `image`, `figure`, and `table`.
`figure_title` may caption either image/figure content or a table. Charts use
image/figure blocks; do not invent chart- or table-caption raw types. Table
HTML/Markdown remains inside its table block.

Process visual objects top-down in DeepSeek reading order. Consecutive visual
objects with the same role join one run when separated by no more than two raw
`text` blocks. Intervening text is ignored for linking and is not added to the
bundle. Images/figures and tables use the same algorithm but are not combined
into one primary run.

For each run, search upward first for an eligible unclaimed `figure_title`,
ignoring at most two raw text blocks. If none exists, search downward under the
same limit. A primary title is claimed once. After primary linking, an
unclaimed floating title connects to every compatible visual whose center is
within 350 normalized-page units. These fallback edges may join multiple runs.

Each connected component becomes `visual_bundle`, preserving actual member
boxes, reading orders, text/table content, and primary-versus-fallback link
provenance. Standalone visuals and orphan titles remain auditable diagnostics.
MinerU may remain a viewer comparison layer but supplies no final candidate or
target in this program.

## Candidate record and revision

Stage 00 materializes one deterministic candidate universe and assigns a
`candidate_revision` over:

- V1 manifest and source hashes;
- DeepSeek parser/code revision;
- section and visual-linking configuration;
- normalization and reading-order rules;
- rendering parameters;
- crop/masked-page representation;
- candidate-type vocabulary; and
- output schema.

The candidate table is reusable and query-independent. Each record retains:

- stable `candidate_id`, `canonical_page_id`, and `candidate_revision`;
- candidate type and raw DeepSeek member types;
- actual member boxes, not only an envelope;
- member reading orders and link provenance;
- OCR, table, formula, caption, or heading content as available;
- rendered crop or masked-page path and SHA-256;
- source page/image/OCR identities and hashes; and
- diagnostics and quarantine flags that do not enter the model.

No query-relative relation, answer string, source family, conflict status,
mapping confidence, gold geometry, or review label is stored as a reusable
candidate feature.

Any change to headings, paragraph fallback, raw-type compatibility, visual-run
formation, caption claiming, geometry fallback, member-union rules, rendering,
or coverage mapping creates a new dataset definition and candidate revision.
Update implementation, tests, architecture, and generated manifests together.

## Geometry and answer-anchor mapping

Candidate geometry is the geometric union of actual member boxes. An enclosing
rectangle may be recorded as metadata but never substitutes for the union when
computing coverage.

For one accepted gold answer-location group `G` and candidate member union `C`:

```text
gold_coverage(G, C) = area(G intersect C) / area(G)
candidate_precision(G, C) = area(G intersect C) / area(C)
```

Binding coverage labels:

- `gold_coverage >= 0.70`: `positive_anchor` for that alternative;
- `0 < gold_coverage < 0.70`: `partial_anchor`;
- `gold_coverage == 0`: non-anchor candidate, not automatically a verified
  negative.

Member boxes that jointly describe one annotated answer are evaluated as a
group. Alternative accepted occurrences are OR-equivalent groups. The program
does not require every alternative occurrence to be selected and does not
flatten them into one mandatory union.

## Eligibility and exclusions

A primary-split question is eligible only when:

- `usable_in_v1=true`;
- the packaged OCR-backed supplied page resolves;
- at least one accepted answer-location group exists on that page;
- no unresolved answer, page, or box disagreement affects the target used by
  the program; and
- at least one candidate reaches `gold_coverage >= 0.70`.

Every ineligible question produces exactly one derived exclusion row. Reasons
include at least:

- `source_not_usable`;
- `unresolved_answer_disagreement`;
- `unresolved_page_disagreement`;
- `unresolved_box_disagreement`;
- `missing_answer_location`;
- `invalid_or_missing_ocr_page`;
- `candidate_generation_failure`;
- `partial_anchor_coverage`;
- `no_anchor_covering_segment`; and
- `no_verified_negative_for_training` where applicable.

The older 4K segmentation pipeline quarantined a page when a question had zero
or multiple positives or any partial relation. This V1 program intentionally
reconciles that legacy rule: accepted alternative positives use OR semantics;
partial and unresolved candidates are masked; only unresolved target conflicts
or missing covering candidates exclude the question. No manual correction is
inserted into generated labels.

## Candidate-oracle report

Candidate performance is a ceiling distinct from selector quality. Stage 00
must report, before any model metric:

- total source, eligible, and excluded questions;
- candidate-oracle hit/recall overall and by split and source;
- candidate count p50, p90, p95, and maximum;
- member count and disconnected-member distributions;
- candidate type;
- OCR quality and failure flags;
- answer-box size and topology;
- single-box versus grouped-box and alternative-occurrence behavior;
- partial-anchor counts;
- unknown type counts;
- member overlap and duplicate-box rates; and
- every exclusion reason.

Historical suggestions of at least 95% overall and 90% strict multi-box oracle
coverage are planning targets, not measured binding thresholds. SOL reports the
actual result; the control plane decides whether it supports the intended
claim. If coverage is poor, repair and refreeze candidate generation under a
new revision before scoring models.

## Verified-negative architecture

Candidate absence from the positive anchor is insufficient to prove
irrelevance. Strong negative construction is training-only and query-relative.

For each eligible training question:

1. retain every accepted positive candidate;
2. remove all positive, partial, or accepted-alternative-overlap candidates
   from the negative pool;
3. remove candidates whose normalized OCR contains an accepted answer string;
4. remove plausible unresolved or unverified context from negative loss;
5. rank remaining same-page candidates with R0;
6. retain up to four highest-scoring `hard_non_anchor` candidates;
7. retain up to four ordinary candidates under stable SHA-256 ordering; and
8. record every selection and removal reason with score, rank, geometry, text,
   type, and revision.

Audit at least 200 deterministic stratified hard candidates across source,
rank, and OCR quality, with at least 20 rows for every represented source
family. Audit outcomes are:

- `valid_hard_non_anchor`;
- `likely_false_negative`; and
- `uncertain`.

Only the first enters verified-negative loss. Report the measured likely false
negative and uncertain rates. Validation, internal test, and holdout rows never
enter mining or optimization.

## Common model-input boundary

Permitted inference-time fields are:

- the original question;
- the supplied clean page or rendered candidate, depending on the arm;
- candidate member geometry;
- candidate OCR/table/formula/caption text;
- candidate type;
- reading order; and
- typed structural relations only if they actually exist in the frozen
  candidate manifest.

Forbidden inputs include:

- gold answer text or accepted answer strings;
- gold boxes or overlap values;
- positive/negative/partial labels;
- dataset/source identity;
- conflict, audit, exclusion, or review status;
- mapping confidence or candidate rank derived from labels; and
- invented parent-child hierarchy.

Answers remain provenance and audit data, never model input.

## Candidate aliases

For the later H0/H1/A1 answer-sufficiency experiment, consecutive numeric
aliases are assigned under deterministic candidate reading order. Aliases are
not built or consumed by the active POC. The later alias map and overlay are a
derived serialization view, not a candidate revision, and must preserve a
round-trip mapping from alias to exact `candidate_id` plus overlay, prompt, and
candidate-list hashes.

Alias colors, box styles, order, omissions, and visibility must not encode
gold or audit metadata. The clean page remains authoritative; OCR/table text is
a hint. Label collisions, unreadable aliases, content occlusion, and list/page
mismatches are measured before any structured-policy training.

## Candidate truncation

Do not choose a candidate cap before measuring candidate-count distributions
and memory at p95. If a cap is required:

- freeze one deterministic inference-time truncation rule;
- apply it identically to all compatible arms;
- emit a new view/config hash;
- report post-truncation candidate-oracle loss overall and by slice; and
- never use gold overlap, answer text, or target status to retain candidates.

## Holdout boundary

`infographicsvqa_holdout` is excluded from candidate-driven model selection,
threshold fitting, early stopping, negative mining, optimization, architecture
promotion, and prompt/reward changes. After Stage 04 is locked, Stage 05 may
build the same candidate/view artifacts and evaluate the frozen chosen systems
once as a robustness test. Holdout results cannot change the chosen model.
