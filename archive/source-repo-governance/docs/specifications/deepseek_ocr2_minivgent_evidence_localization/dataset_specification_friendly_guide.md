# Dataset Specification for Controlled DeepSeek-OCR2 and MiniVGent Evidence-Localization Experiments

## Friendly Research Guide Edition

This edition keeps the full research specification intact, but gives the reader a more welcoming path through it. Think of it as the same laboratory notebook with clearer labels on the drawers.

Nothing substantive has been removed. The dataset decisions, counts, percentages, tables, citations, experimental arms, schemas, risks, unresolved questions, and owner-approval boundaries remain part of the specification.

### A tiny jargon map before we begin

- **Evidence selector:** the component that chooses the page regions needed to answer a question.
- **Candidate universe:** the complete, frozen list of regions the selector is allowed to choose from.
- **SFT (supervised fine-tuning):** training from examples that already contain the desired output.
- **Autoregressive selection:** choosing outputs one after another, like writing a list from left to right.
- **Parallel set prediction:** scoring all candidate regions together and returning the relevant set.
- **Oracle recall:** whether the correct evidence was available in the candidate universe at all, even before judging the selector.
- **Quarantined or sealed evaluation:** data kept completely out of training, prompt tuning, threshold selection, and error-driven data generation.
- **Complete evidence set:** the smallest audited collection that is sufficient to derive the answer and where every required role is genuinely necessary.

### How to read the confidence labels

The original machine-readable label stays beside every friendlier phrase:

- **What the official source says** (`primary_source_fact`)
- **What we directly verified in the release** (`verified_release_observation`)
- **What we calculated from cited facts** (`derived_calculation`)
- **Our evidence-based interpretation** (`researcher_inference`)
- **Our proposed plan** (`proposed_design`)
- **Still unresolved** (`unresolved`)

**Status:** Nonbinding future experiment proposal; no model training, dataset acquisition, or repository modification is authorized or represented as completed.<br>
**Research cutoff:** 2026-08-06.<br>
**Access date for all web sources unless otherwise stated:** 2026-08-06.
**Active-project boundary:** the separate active implementation remains limited to Visual-CoT acquisition and normalization through its dataset-foundation stages. This report does not amend or replace that implementation contract.
**Owner corpus amendment (2026-08-07):** TextVQA and TextCaps are audit-only exclusions. They contribute zero records to selector SFT, augmentation, planner/answerer training, RL, calibration, and evaluation. Their release counts and audit cards remain only for source accounting and overlap control.

---


## 1. The short version — our main decisions

**Friendly guidepost.** If you read only one section first, make it this one. It tells you which datasets carry the plan, how large the two proposed corpora are, what is deliberately excluded, and which fairness checks must pass before any model comparison means much.

### 1.1 What we recommend

**Our proposed plan** (`proposed_design`) Build the selector corpus around **five evidence-bearing anchors** and keep weaker or contaminated sources outside selector SFT:

1. **Visual-CoT document-derived records** from DocVQA, InfographicVQA, and SROIE, with the original parent question retained and every box treated as *answer-bearing pseudo-evidence* until complete-evidence verification passes. The paper reports rounded counts (26k DocVQA, 20k InfographicVQA, 15k DUDE, 4k SROIE, 18k TextVQA, 32k TextCaps), while the pinned GitHub snapshot contains exactly **26,117 / 19,429 / 6,088 / 2,584 / 17,280 / 33,866** JSONL rows, respectively. The pinned files, not the rounded paper table, govern quotas. The TextVQA and TextCaps files are retained in this inventory only to reproduce and audit the release; they contribute zero project records. ([VCOT-P, §3.2 and Table 2][VCOT-P]; [VCOT-GH])
2. **Original DUDE training records**, because DUDE supplies real multi-page documents, extractive/abstractive/list/unanswerable questions, and native evidence boxes for extractive answers. Published totals conflict between the abstract and Table 2, so the release manifest must be treated as authoritative during acquisition. ([DUDE-P, abstract and Table 2][DUDE-P])
3. **TAT-DQA**, because it retains TAT-QA-style numerical programs and evidence while restoring page images, OCR, layout, and document structure. It supplies 13,251 train, 1,645 development, and 1,662 test questions over 3,067 pages. ([TATDQA-P, §3 and Table 2][TATDQA-P])
4. **QASPER and PeerQA after PDF reacquisition and deterministic paragraph-to-candidate alignment**, because they provide realistic paper-reader or peer-review questions with textual evidence rather than answer-string boxes. QASPER has 2,593 train questions over 888 train papers; PeerQA has 579 questions over 208 papers but no official train/development/test split. ([QASPER-P, Table 1][QASPER-P]; [PEERQA-P, §3 and Table 2][PEERQA-P])
5. **RefChartQA after license review**, because it supplies chart-element boxes and reference-resolution questions. The current official release reports 55,789 train, 6,223 validation, and 11,690 test records; it is not assumed to annotate every header, legend, or scale dependency required for a complete evidence set. ([REFCHART-GH][REFCHART-GH]; [REFCHART-HF][REFCHART-HF])

**Our proposed plan** (`proposed_design`) Start with the selected **Visual-CoT DocVQA, InfographicVQA, and SROIE views**, then use **BoundingDocs v2.0 only as source-distinct auxiliary grounding data**. Visual-CoT is the starter because those source-specific files have pinned row counts, explicit parent/source labels, a narrow common schema, and a smaller provenance surface that can be grouped and audited before broad mixture sampling. BoundingDocs is much larger, but its unified release mixes heterogeneous parents, overlaps DUDE/MP-DocVQA/SP-DocVQA, inherits multiple parent licenses, and requires independent reproduction of the v2 alignment fixes plus source-balanced sampling. This is a provenance and experimental-control decision, not a claim that Visual-CoT boxes are more complete: both sources generally localize answer-bearing text and remain Level C until necessity and sufficiency verification passes. ([BOUND-P, §3][BOUND-P]; [BOUND-HF][BOUND-HF])

### 1.2 How large the training collections will be

- **Our proposed plan** (`proposed_design`) **Minimum decisive selector corpus:** 32,000 unique SFT records: 18,000 accepted original records and 14,000 accepted augmentations.
- **Our proposed plan** (`proposed_design`) **Full recommended selector corpus:** 128,000 unique SFT records: 64,000 accepted original records and 64,000 accepted augmentations.
- **Our proposed plan** (`proposed_design`) **Additional full-plan pools:** 32,000 planner/decomposition records, 16,000 answerer-SFT records, and a conditional 20,000-record later-RL pool. These pools are not selector labels unless they separately satisfy the Level-C pseudo-evidence contract.

### 1.3 What stays out of training — and why

- **What we directly verified in the release** (`verified_release_observation`) **BBox DocVQA is not counted as a separate current dataset.** It is arXiv:2511.15090v1; v2 was renamed and substantially redefined as **SciEGQA**. ([BBOX-V1][BBOX-V1]; [SCIEGQA-P][SCIEGQA-P])
- **Our proposed plan** (`proposed_design`) **SciEGQA-Train is excluded pending an explicit dataset license and independent verification of its automatically generated evidence.** SciEGQA-Bench is sealed evaluation only. ([SCIEGQA-TRAIN][SCIEGQA-TRAIN]; [SCIEGQA-BENCH][SCIEGQA-BENCH])
- **What we directly verified in the release** (`verified_release_observation`) **GroundingDocQA is unavailable for acquisition as of the cutoff.** The official project reports 200k documents and 2M QA pairs but labels the dataset “Soon” and exposes no usable license or release artifact. ([M3G-PROJ][M3G-PROJ])
- **What we directly verified in the release** (`verified_release_observation`) **SlideVQA is evaluation-only under its repository license.** The license permits internal/evaluation use and does not authorize training or derivative-data generation. ([SLIDE-LIC][SLIDE-LIC])
- **Our proposed plan** (`proposed_design`) **MMLongBench-Doc and MMLongBench-Doc-V2 are sealed external benchmarks.** V2, released 2026-08-04, is the primary future external score; V1 remains historical. No question, evidence, correction log, prompt, or error analysis from either release may affect training, thresholds, prompts, or dataset selection. ([MMLBD-P][MMLBD-P]; [MMLBD2-P][MMLBD2-P])
- **Our proposed plan** (`proposed_design`) M-LongDoc, LongDocURL, M3DocVQA, MMDocIR expert, MMDocRAG, DocScope, and ChartQAPro are also quarantined as external or unresolved benchmarks.

### 1.4 How we make the architecture comparison fair

**What the official source says** (`primary_source_fact`) DeepSeek-OCR-2 is an approximately 3.39B-parameter OCR-focused vision-language model released under Apache-2.0. Qwen3-VL-Reranker-2B is an approximately 2.13B-parameter multimodal reranker initialized from Qwen3-VL-2B-Instruct, also under Apache-2.0. ([DSOCR2-HF][DSOCR2-HF]; [QWEN-RERANK-HF][QWEN-RERANK-HF])

**Our proposed plan** (`proposed_design`) The primary MiniVGent arm therefore uses **Qwen3-VL-Reranker-2B or a separately approved reranker-class backbone**, not DeepSeek-OCR2. DeepSeek-OCR2 is limited to Arms A/B and the frozen offline OCR/semantic-candidate pipeline.

**Our evidence-based interpretation** (`researcher_inference`) Arm B versus Arm C cannot by itself isolate autoregressive ID generation from parallel set selection because the backbone also changes. The decisive study therefore adds a mandatory diagnostic:

- **Arm Bq: Qwen-AR-ID** — the same Qwen reranker initialization, images, metadata, candidates, and training records as MiniVGent-Set, but with an autoregressive candidate-ID decoder.

The clean contrasts are:

| Contrast | Interpretable causal question |
|---|---|
| A vs B | Does raw coordinate generation under DeepSeek-OCR2 hurt relative to selecting frozen candidate IDs? |
| Bq vs C | Does parallel set prediction help relative to autoregressive ID generation under the same Qwen initialization? |
| B vs C | Which full system is better, acknowledging both backbone and decoder differences? |
| A vs C | Which deployment architecture is better end to end, not why? |

### 1.5 Checks that must pass before we trust the results

**Our proposed plan** (`proposed_design`) No MiniVGent result is interpretable until the frozen candidate universe reaches:

- atomic evidence oracle recall ≥ 98.5% overall;
- strict all-required-evidence oracle recall ≥ 95.0% overall;
- strict all-required-evidence oracle recall ≥ 92.0% on genuine cross-page records;
- strict all-required-evidence oracle recall ≥ 90.0% on chart/header dependency records;
- candidate count p95 ≤ 512 per page and ≤ 2,048 per multi-page bundle.

Failure triggers parser/candidate revision, not selector blame.

---


## 2. What we are testing — and how we keep it fair

**Friendly guidepost.** Think of this section as the rules of the tournament. It defines exactly what each model sees, what each model must return, and which conditions stay fixed so that no arm receives an accidental advantage.

### 2.1 What one training or evaluation record contains

A record consists of a natural-language question, one real document or explicitly labeled synthetic bundle, page images, OCR/layout metadata, a frozen candidate universe, and a gold evidence set. The selector never receives the gold answer, gold evidence roles, evidence boxes, programs, source-specific answer metadata, or verifier outputs.

A **complete evidence set** is the smallest audited set of regions/pages that satisfies both:

1. **Sufficiency:** the answer is derivable from the complete set under the record’s declared reasoning program; and
2. **Necessity:** each required evidence role fails a leave-one-role-out answerability test. Multiple equivalent regions may be represented as OR-groups; conjunctive requirements are represented as AND-groups.

An answer-bearing span is not automatically a complete evidence set. A page label is not a region label. A derivation program is not spatial supervision unless every operand and structural dependency is aligned.

### 2.2 The experimental arms

#### Arm A — DeepSeek-OCR2-Box

- Input: page image or page bundle plus question.
- Output: an autoregressively generated, canonicalized set of boxes; empty set is a first-class valid output.
- Training target: normalized boxes sorted by page and reading order, with set-equivalent evaluation independent of serialization order.
- Purpose: test flexible coordinate generation without a proposal ceiling.

#### Arm B — DeepSeek-OCR2-ID

- Input: identical images/question plus the exact frozen candidate universe supplied to Arm C.
- Output: autoregressively generated candidate IDs; empty list is valid.
- Purpose: control the action space while retaining DeepSeek-OCR2 autoregression.

#### Arm C — MiniVGent-Set

- Input: identical images/question/candidates/OCR/layout metadata and only the question-conditioned backbone features allowed by the frozen protocol.
- Backbone: Qwen3-VL-Reranker-2B by default; another reranker-class backbone requires owner approval and a new protocol version.
- Output: parallel candidate membership probabilities, explicit empty-set/abstention probability, and optional cardinality/sufficiency heads.
- Purpose: test parallel set-aware evidence selection.

#### Mandatory diagnostic Arm Bq — Qwen-AR-ID

- Input and initialization: identical to Arm C.
- Output: autoregressive candidate IDs.
- Purpose: isolate autoregression versus parallel set prediction without changing the backbone.

### 2.3 What must remain identical across arms

All primary arms receive the same:

- visual-identity groups and document splits;
- original questions and accepted augmentations;
- source revisions and renderings;
- evidence semantics and ignored-candidate masks;
- candidate universe for B, Bq, and C;
- answerer checkpoint, answer prompt, decoding parameters, and selected-evidence formatting;
- selection budgets expressed as both candidate count and selected pixel area/token cost;
- internal development/test partitions and external sealed benchmarks;
- stopping rules and threshold-selection data.

The DeepSeek semantic parser is run once, versioned, and frozen. No selector loss updates the parser. Candidate revisions create a new experimental protocol and invalidate direct comparison to earlier runs.

### 2.4 Find the evidence first; answer the question second

Evidence localization is measured before answer generation. The answerer is then run on:

1. gold evidence;
2. each arm’s selected evidence only;
3. full retrieved pages under the same page-retrieval gate; and
4. leave-one-gold-role-out evidence.

No arm may generate an answer in lieu of selecting evidence. Answer quality is a downstream consequence, not the localization label.

---


## 3. How we researched and checked the evidence

**Friendly guidepost.** This is the report's trust layer. It explains which sources count as evidence, how claims are labeled, and what happens when a paper, repository, and released data disagree.

### 3.1 How each source was checked

**Our proposed plan** (`proposed_design`) The audit used, where available, the complete paper and appendices, official project page, official repository, dataset card, file tree/schema, release metadata, and license. Search snippets, blogs, leaderboards, and third-party mirrors were not accepted as the sole basis for counts or inclusion.

For every source, the ledger records the item inspected, release/tag/commit when visible, and access date. Where paper and release disagree, both values are retained and the acquisition manifest decides the operative count.

### 3.2 How to read the confidence labels

**Audit-card citation convention.** In every 28-field card, row 2 names the controlling primary paper and official release. Those sources govern the factual rows immediately below; row-level source IDs are repeated wherever a number, schema detail, release observation, conflict, or example requires a narrower locator. This keeps citations adjacent without repeating the same two links in all 28 cells.

- **What the official source says** (`primary_source_fact`) — explicitly stated in a paper, official page, official repository, or data card.
- **What we directly verified in the release** (`verified_release_observation`) — directly observed in a released file tree, schema, license, revision, or row metadata.
- **What we calculated from cited facts** (`derived_calculation`) — arithmetic over cited source facts.
- **Our evidence-based interpretation** (`researcher_inference`) — interpretation supported by cited facts but not directly stated by a source.
- **Our proposed plan** (`proposed_design`) — this project’s future protocol or threshold.
- **Still unresolved** (`unresolved`) — evidence unavailable or contradictory at the cutoff.

### 3.3 What happens when sources disagree

1. Preserve every conflicting value.
2. State what each value appears to count.
3. Use the release manifest for acquisition if the release is available; otherwise use the paper only as a planning ceiling.
4. Never convert rounded “k” counts into invented exact counts.
5. Never infer a dataset license from a code license.
6. Apply the most restrictive applicable terms when parent images, annotations, and code use different licenses.

### 3.4 How strong is each kind of evidence supervision?

| Level | Definition | Selector use |
|---|---|---|
| A | Native gold boxes, polygons, cells, masks, or grounded regions that jointly cover all required evidence roles | Direct selector SFT; still audit completeness |
| B | Native supporting paragraphs, pages, cells, facts, operands, or programs that can be deterministically aligned | Selector SFT after alignment/oracle checks |
| C | Pseudo-evidence produced by a separate generator and accepted by independent sufficiency, necessity, consistency, and human-audit checks | Selector SFT with provenance and confidence |
| D | Question/answer only, or evidence too incomplete to trust | Answerer SFT or later outcome-based RL only |

The same dataset may yield different levels record by record.

---


## 4. A practical guide to question, evidence, and supervision types

**Friendly guidepost.** The vocabulary gets dense here, so this section works like a field guide. It gives every record a consistent answer type, evidence shape, reasoning operation, modality relationship, and supervision strength.

### 4.1 What kind of answer is expected?

Every record receives exactly one primary answer type:

- `extractive_span`
- `list_or_multi_span`
- `yes_no`
- `numerical_or_program_derived`
- `semi_extractive_synthesis`
- `abstractive_synthesis`
- `unanswerable_or_insufficient_evidence`

Aliases and acceptable numeric tolerances are stored separately.
**In ordinary language:**

- `extractive_span`: copy one answer span from the source.
- `list_or_multi_span`: return several separate items or spans.
- `yes_no`: answer yes or no.
- `numerical_or_program_derived`: calculate the answer from values or a reasoning program.
- `semi_extractive_synthesis`: combine quoted facts with a small amount of connecting language.
- `abstractive_synthesis`: explain or synthesize the answer rather than copying it directly.
- `unanswerable_or_insufficient_evidence`: the document does not contain enough evidence for a supported answer.

### 4.2 Where is the evidence located?

- `zero_evidence`
- `one_region`
- `multiple_regions_one_page`
- `multiple_pages_real_document`
- `multiple_documents`
- `synthetic_bundle_cross_page`
- `global_or_exhaustive_document_scope`

A question is labeled multi-hop only when one evidence role is conditionally required to identify, interpret, transform, or constrain another. Two independent lookups joined by “and” are recorded as composition, not genuine dependency.
**A quick picture of the evidence shapes:**

- `zero_evidence`: the correct result is an empty set.
- `one_region`: one page region is enough.
- `multiple_regions_one_page`: several regions on the same page are required.
- `multiple_pages_real_document`: the evidence is spread across pages of one real document.
- `multiple_documents`: more than one real document is required.
- `synthetic_bundle_cross_page`: pages were intentionally bundled to create a controlled training example.
- `global_or_exhaustive_document_scope`: the question requires searching or counting across the document as a whole.

### 4.3 What kind of thinking does the question require?

The controlled vocabulary is:

`direct_lookup`, `comparison`, `arithmetic`, `aggregation`, `temporal_reasoning`, `condition_plus_exception`, `definition_plus_application`, `entity_resolution`, `table_header_or_row_column_dependency`, `chart_plus_caption_or_legend`, `claim_plus_qualification`, `contradiction_or_reconciliation`, `causal_or_explanatory_synthesis`, `document_wide_counting_or_exhaustive_search`, `missing_premise_detection`, and `document_level_synthesis`.

Secondary operations may be attached, but every record has one primary operation for stratification.
**Plain-language examples of those operations:**

- `direct_lookup`: find a stated fact.
- `comparison`: decide how two or more items differ.
- `arithmetic`: calculate with values from the document.
- `aggregation`: combine several values or instances.
- `temporal_reasoning`: reason about dates, order, or change over time.
- `condition_plus_exception`: apply a rule while respecting its exception.
- `definition_plus_application`: find a definition and use it in context.
- `entity_resolution`: work out which person, organization, item, or mention is the same entity.
- `table_header_or_row_column_dependency`: interpret a cell using its headers or table position.
- `chart_plus_caption_or_legend`: combine the chart with the caption, labels, scale, or legend.
- `claim_plus_qualification`: pair a claim with the caveat that limits it.
- `contradiction_or_reconciliation`: resolve apparently conflicting statements.
- `causal_or_explanatory_synthesis`: explain why something happened or what it means.
- `document_wide_counting_or_exhaustive_search`: inspect the full document rather than stopping at the first match.
- `missing_premise_detection`: notice that a required fact is absent.
- `document_level_synthesis`: combine evidence spread across the broader document into one supported answer.

### 4.4 Which visual or textual pieces must work together?

`text_to_text`, `table_to_text`, `table_to_table`, `chart_to_text`, `chart_to_caption_or_legend`, `figure_to_caption`, `cell_to_header`, `paragraph_to_footnote`, `cross_page_text`, and `mixed_visual_textual_evidence`.
**In ordinary language:** these labels tell us which pieces must be connected — text with text, tables with prose or other tables, charts with labels or captions, cells with headers, paragraphs with footnotes, text across pages, or a mixture of visual and textual evidence.

### 4.5 How the required evidence pieces fit together

Evidence is represented as a Boolean expression over atomic roles:

```text
AND(role_1, role_2, OR(role_3a, role_3b))
```

This permits equivalent renderings or repeated headers without treating every equivalent candidate as a false negative. Candidates satisfying an accepted OR-alternative are positive; candidates whose relevance is unresolved are `ignore`, not negative.

---


## 5. Datasets with extractive answers or spatial evidence

**Friendly guidepost.** These are the sources that most directly point to where an answer appears on a page. A box around an answer is useful, but it is not automatically the full evidence needed to justify that answer — that distinction matters throughout the audit cards. Each 28-row card follows the same path: identity and access (rows 1-7), size and splits (8-11), question and evidence structure (12-20), risks and examples (21-22), then the project decision and remaining work (23-28).

This section audits every required spatial/grounding seed. `Planned original records` are project targets, not published facts. A value of zero is an explicit decision, not missing work.

### 5.1 Visual-CoT

> **Quick take**
> - **Main project role:** `core_selector_sft` for the DocVQA, InfographicVQA, and SROIE views; DUDE, TextVQA, and TextCaps views are excluded.
> - **What it uniquely adds:** Provides the controlled Level-C starter through pinned, source-explicit DocVQA, InfographicVQA, and SROIE views. TextVQA and TextCaps add no project records.
> - **Planned original records:** **Our proposed plan** (`proposed_design`) Minimum: 5,500 source-distinct document records (3,000 DocVQA; 2,000 InfographicVQA; 500 SROIE). Full: 27,000 (**16,000 DocVQA; 9,000 InfographicVQA; 2,000 SROIE**). DUDE, TextVQA, and TextCaps derivatives: 0.
> - **Why this decision:** It is the controlled starter because its selected source views are pinned, source-explicit, and simpler to audit than the BoundingDocs mixture. Its boxes are still pseudo-evidence, not intrinsically stronger or more complete than BoundingDocs boxes.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning**; release aliases `Visual-CoT`, `viscot_dataset`. |
| 2 | Where to find the paper and official release | Paper and appendix: [VCOT-P]. Official code/data repository: [VCOT-GH]. Official Hugging Face release: [VCOT-HF]. |
| 3 | Can we get it now? | **What we directly verified in the release** (`verified_release_observation`) Public metadata and image archives are available. The repository exposes source-specific JSONL files rather than a single independent document collection. |
| 4 | Exact version or revision | **What we directly verified in the release** (`verified_release_observation`) Paper arXiv v3. GitHub snapshot fixed to `83212ae474ab46b70048a6df6a386cff6589ff92`; HF snapshot observed at `223d2d8c1146fda2bb918801b8276c587b78b61c`. [VCOT-P] [VCOT-GH] [VCOT-HF] |
| 5 | What the license allows — and does not allow | **Still unresolved** (`unresolved`) The HF UI and card text expose inconsistent licensing signals (Apache-2.0 versus CC BY-NC 4.0/research-only). Parent-image and parent-annotation terms remain controlling. Use only after a written license matrix; never redistribute parent images under the code license. [VCOT-HF] [VCOT-GH] |
| 6 | Where the original documents came from | **What the official source says** (`primary_source_fact`) Derivative records from DocVQA, InfographicVQA, DUDE, SROIE, TextVQA, TextCaps, and several non-document vision datasets. [VCOT-P] |
| 7 | What kind of material it contains | Page images, infographics, receipts, and scene-text images paired with text questions, answers, and bounding boxes. |
| 8 | How many documents | **What the official source says** (`primary_source_fact`) Not a new document count; documents/images are inherited from parents. Counting Visual-CoT and parents as separate documents would double-count visual identities. [VCOT-P] |
| 9 | How many pages or images | **What the official source says** (`primary_source_fact`) Paper Table 2 reports rounded source-example counts, not unique image totals. |
| 10 | How many questions | **What the official source says** (`primary_source_fact`) Paper Table 2 gives rounded counts: DocVQA 26k, InfographicVQA 20k, DUDE 15k, SROIE 4k, TextVQA 18k, TextCaps 32k. **What we directly verified in the release** (`verified_release_observation`) At GitHub commit `83212ae474ab46b70048a6df6a386cff6589ff92`, exact JSONL rows are **26,117; 19,429; 6,088; 2,584; 17,280; and 33,866**, respectively. [VCOT-P] [VCOT-GH] |
| 11 | Official train, development, and test counts | **What we directly verified in the release** (`verified_release_observation`) Source JSONL rows retain parent `split`; this project uses only parent-train records and re-groups by visual identity before augmentation. [VCOT-GH] |
| 12 | Kinds of answers | Predominantly extractive short answers; TextCaps-derived questions are synthetic; DUDE includes list/abstractive/unanswerable parents but the released Visual-CoT selection is not a faithful complete-evidence view. |
| 13 | Kinds of reasoning | Direct lookup, OCR reading, simple spatial grounding, limited multi-span lookup; no guarantee of genuine dependency. |
| 14 | Where and how the evidence is spread | Mostly one region; some rows contain repeated or multiple answer boxes; DUDE parent images may be multi-page, but the derivative row is page-image grounded. |
| 15 | What evidence labels exist and how precise they are | **What we directly verified in the release** (`verified_release_observation`) Schema includes `question`, `answer`, `possible_answers`, `image`, `width`, `height`, `bboxs`, `dataset`, and `split`. Boxes were built by matching answer strings to OCR/layout. [VCOT-GH] |
| 16 | How spatial coordinates are represented | Absolute pixel `[x1,y1,x2,y2]` in the released image dimensions. |
| 17 | Are the original PDFs or page images available? | Images are released through source-specific archives, subject to parent terms. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | High for boxes that overlap parser text units; unmatched or partial answer boxes require snapping with an IoU/text-overlap audit. Multi-box rows require role reconstruction. |
| 19 | Does it label the complete evidence set? | **No, not generally.** Boxes usually identify answer-bearing text, not every operand, header, qualifier, legend, definition, or exception needed to answer. |
| 20 | Known gaps or quality concerns | Answer-string matching can duplicate boxes, omit structural dependencies, or select a coincidental repeated string. TextCaps questions are generated rather than user-authored. |
| 21 | Could it overlap with other training or evaluation data? | Hard overlap with every parent dataset; Visual-CoT DUDE must not coexist with original DUDE. Parent train/dev/test groups control all derivatives. |
| 22 | Real example questions from official material | Official released rows include: “what is the contact person name mentioned in letter?” → “P. Carter”; “Which corporation's letterhead is this?” → “Brown & Williamson Tobacco Corporation”; “How many car crashes are caused by texting every year?” → “1.6 million.” [VCOT-GH] |
| 23 | What this source uniquely adds | Provides the controlled Level-C starter through pinned, source-explicit DocVQA, InfographicVQA, and SROIE views. TextVQA and TextCaps add no project records. |
| 24 | How many original records this project plans to use | **Our proposed plan** (`proposed_design`) Minimum: 5,500 source-distinct document records (3,000 DocVQA; 2,000 InfographicVQA; 500 SROIE). Full: 27,000 (**16,000 DocVQA; 9,000 InfographicVQA; 2,000 SROIE**). DUDE, TextVQA, and TextCaps derivatives: 0. |
| 25 | Planned augmentations and exact quotas | **Our proposed plan** (`proposed_design`) Minimum accepted derivatives: DocVQA 2,000; InfographicVQA 1,800; SROIE 500. Full accepted derivative quotas are in the augmentation accounting matrix; no more than one positive paraphrase per parent and no derivative may cross splits. |
| 26 | Its main job in this project — or the decision to exclude it | `core_selector_sft` for the DocVQA, InfographicVQA, and SROIE views; DUDE, TextVQA, and TextCaps views are excluded. |
| 27 | Why that decision makes sense | It is the controlled starter because its selected source views are pinned, source-explicit, and simpler to audit than the BoundingDocs mixture. Its boxes are still pseudo-evidence, not intrinsically stronger or more complete than BoundingDocs boxes. |
| 28 | What still must be acquired or verified | Resolve the license conflict, verify every parent split and parent-image identity, reproduce the exact pinned-file line counts, and measure complete-evidence acceptance by source before any final manifest is frozen. |

**What we directly verified in the release** (`verified_release_observation`) **Pinned-file counting method.** Each source file is newline-delimited JSON. The release observation above is the raw line count after confirming every nonblank line parses as one JSON object; it is not a count of unique pages, documents, or visual identities. Acquisition must reproduce the six SHA-pinned line counts before filtering.

### 5.2 Visual-CoT “document versions” / document-source views

> **Quick take**
> - **Main project role:** `exclude` as an independent dataset identity.
> - **What it uniquely adds:** Provides a convenient normalized schema, not an independent data source.
> - **Planned original records:** 0 as a separate corpus identity; all quotas are counted under explicit source views in §5.1.
> - **Why this decision:** Prevents duplicate accounting and false claims of a new document collection.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | No separate canonical dataset. This phrase denotes the DocVQA, InfographicVQA, DUDE, and SROIE files inside Visual-CoT. |
| 2 | Where to find the paper and official release | Use [VCOT-P] and the pinned source JSONL directory in [VCOT-GH]. |
| 3 | Can we get it now? | Public as source-specific JSONL and image archives. |
| 4 | Exact version or revision | Same pinned Visual-CoT revisions as §5.1. |
| 5 | What the license allows — and does not allow | Parent-specific terms plus the most restrictive Visual-CoT release terms. |
| 6 | Where the original documents came from | Inherited parent images; no new document acquisition. |
| 7 | What kind of material it contains | Document pages, infographics, receipts; the files are views, not independent corpora. |
| 8 | How many documents | Not separately countable without parent visual-identity hashes. |
| 9 | How many pages or images | Not separately countable without parent visual-identity hashes. |
| 10 | How many questions | Rounded rows are reported in Visual-CoT Table 2. |
| 11 | Official train, development, and test counts | Parent split field is preserved. |
| 12 | Kinds of answers | As in each parent view. |
| 13 | Kinds of reasoning | As in each parent view; predominantly lookup. |
| 14 | Where and how the evidence is spread | One-page/one-or-more answer boxes. |
| 15 | What evidence labels exist and how precise they are | Answer-matched boxes. |
| 16 | How spatial coordinates are represented | Absolute pixels. |
| 17 | Are the original PDFs or page images available? | Available only through parent/Visual-CoT asset terms. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Same as §5.1. |
| 19 | Does it label the complete evidence set? | No. |
| 20 | Known gaps or quality concerns | The name can mislead users into believing these are document-level versions with complete multi-page evidence. |
| 21 | Could it overlap with other training or evaluation data? | Complete overlap with parent questions/images. |
| 22 | Real example questions from official material | See the authentic source-view examples in §5.1 and the parent cards below. |
| 23 | What this source uniquely adds | Provides a convenient normalized schema, not an independent data source. |
| 24 | How many original records this project plans to use | 0 as a separate corpus identity; all quotas are counted under explicit source views in §5.1. |
| 25 | Planned augmentations and exact quotas | 0 separate augmentations; derivatives retain the explicit parent record IDs. |
| 26 | Its main job in this project — or the decision to exclude it | `exclude` as an independent dataset identity. |
| 27 | Why that decision makes sense | Prevents duplicate accounting and false claims of a new document collection. |
| 28 | What still must be acquired or verified | Acquisition code must require `source_dataset` and `parent_record_id`; generic `visual_cot_document` labels are rejected. |

### 5.3 DocVQA (single-page task)

> **Quick take**
> - **Main project role:** `exclude` from independent allocation.
> - **What it uniquely adds:** Parent provenance and page images for Visual-CoT-derived Level-C grounding.
> - **Planned original records:** 0 direct. Minimum/full DocVQA-derived quotas are counted only under Visual-CoT: 3,000/16,000.
> - **Why this decision:** Using both original and derivative questions would duplicate questions and visuals while adding no stronger selector label.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **DocVQA: A Dataset for VQA on Document Images**; commonly SP-DocVQA when contrasted with MP-DocVQA. |
| 2 | Where to find the paper and official release | Paper: [DOCVQA-P]. Official challenge/data portal: [DOCVQA-SITE]. |
| 3 | Can we get it now? | Available through the Robust Reading Competition portal with registration. |
| 4 | Exact version or revision | Original DocVQA task release; no project-specific re-release is authorized. |
| 5 | What the license allows — and does not allow | RRC/data-use terms; the paper does not provide a simple permissive dataset license. Source images derive from the UCSF Industry Documents Library. |
| 6 | Where the original documents came from | UCSF tobacco-industry document pages, primarily 1960–2000. |
| 7 | What kind of material it contains | Single-page scanned/typed/handwritten industry documents with OCR. |
| 8 | How many documents | **What the official source says** (`primary_source_fact`) 12,767 document images. [DOCVQA-P] |
| 9 | How many pages or images | 12,767 single-page images. |
| 10 | How many questions | **What the official source says** (`primary_source_fact`) 50,000 questions. [DOCVQA-P] |
| 11 | Official train, development, and test counts | 39,463 train; 5,349 validation; 5,188 test. [DOCVQA-P] |
| 12 | Kinds of answers | Short extractive answers and aliases. |
| 13 | Kinds of reasoning | Direct lookup, entity/date/value retrieval, occasional layout/table lookup. |
| 14 | Where and how the evidence is spread | One page; answer can be one span but complete contextual evidence is not annotated. |
| 15 | What evidence labels exist and how precise they are | Question/answer plus OCR tokens and OCR boxes; no native complete-evidence set. |
| 16 | How spatial coordinates are represented | OCR provider coordinates; challenge releases must be normalized per OCR file. |
| 17 | Are the original PDFs or page images available? | Page images and OCR available under RRC terms. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | High for OCR-aligned candidates; answer aliases still require text normalization. |
| 19 | Does it label the complete evidence set? | No. |
| 20 | Known gaps or quality concerns | OCR boxes are not gold evidence boxes. The same answer string can occur multiple times; headers and qualifiers are often omitted. |
| 21 | Could it overlap with other training or evaluation data? | Parent of Visual-CoT DocVQA and present indirectly in BoundingDocs/SP-DocVQA-derived data. |
| 22 | Real example questions from official material | Official paper/release examples include the source questions reproduced in Visual-CoT: “what is the contact person name mentioned in letter?” and “Which corporation's letterhead is this?” [VCOT-GH] |
| 23 | What this source uniquely adds | Parent provenance and page images for Visual-CoT-derived Level-C grounding. |
| 24 | How many original records this project plans to use | 0 direct. Minimum/full DocVQA-derived quotas are counted only under Visual-CoT: 3,000/16,000. |
| 25 | Planned augmentations and exact quotas | No direct augmentation stream; all derivatives link to the Visual-CoT row and DocVQA image hash. |
| 26 | Its main job in this project — or the decision to exclude it | `exclude` from independent allocation. |
| 27 | Why that decision makes sense | Using both original and derivative questions would duplicate questions and visuals while adding no stronger selector label. |
| 28 | What still must be acquired or verified | Confirm the RRC data-use agreement and page/question IDs before materialization; compute overlap against BoundingDocs. |

### 5.4 InfographicVQA

> **Quick take**
> - **Main project role:** `exclude` from independent allocation.
> - **What it uniquely adds:** Introduces chart, dense-layout, legend, and infographic diversity.
> - **Planned original records:** 0 direct; use Visual-CoT derivative: minimum 2,000, full 9,000.
> - **Why this decision:** The original release lacks complete spatial evidence; the derivative provides the actionable label while preserving parent provenance.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **InfographicVQA**; also written InfographicsVQA in some derivative files. |
| 2 | Where to find the paper and official release | Paper: [INFO-P]. Official portal: [INFO-SITE]. |
| 3 | Can we get it now? | Available through the DocVQA/RRC portal. |
| 4 | Exact version or revision | Original benchmark release; Visual-CoT uses a source-specific derivative view. |
| 5 | What the license allows — and does not allow | RRC/data-use terms and underlying web-image rights; no simple universal permissive license was verified. |
| 6 | Where the original documents came from | Web infographics spanning diverse topics and designs. |
| 7 | What kind of material it contains | Single infographic images with OCR, charts, icons, legends, and text. |
| 8 | How many documents | **What the official source says** (`primary_source_fact`) 5,485 images. [INFO-P] |
| 9 | How many pages or images | 5,485 images. |
| 10 | How many questions | **What the official source says** (`primary_source_fact`) 30,035 questions. [INFO-P] |
| 11 | Official train, development, and test counts | 23,946 train; 2,801 validation; 3,288 test. [INFO-P] |
| 12 | Kinds of answers | Extractive, list, counting, arithmetic, and visually grounded answers. |
| 13 | Kinds of reasoning | OCR lookup, comparison, arithmetic, chart/legend reading, aggregation. |
| 14 | Where and how the evidence is spread | One image; potentially several regions even when only the answer text is annotated by a derivative. |
| 15 | What evidence labels exist and how precise they are | Question/answer and OCR; no native complete evidence boxes. |
| 16 | How spatial coordinates are represented | OCR coordinates; Visual-CoT derivative uses absolute pixels. |
| 17 | Are the original PDFs or page images available? | Images/OCR available through portal terms. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Moderate: parser candidates must preserve chart/legend, labels, and reading order; OCR-only units are insufficient for many questions. |
| 19 | Does it label the complete evidence set? | No. |
| 20 | Known gaps or quality concerns | Visual-CoT answer boxes can omit axes, legends, denominator values, or comparison operands. |
| 21 | Could it overlap with other training or evaluation data? | Parent of Visual-CoT InfographicVQA; chart imagery may overlap ChartQA-like web sources, requiring perceptual hashing. |
| 22 | Real example questions from official material | Official Visual-CoT source rows include “How many car crashes are caused by texting every year?” and “How many victims come from low- and middle-income communities?” [VCOT-GH] |
| 23 | What this source uniquely adds | Introduces chart, dense-layout, legend, and infographic diversity. |
| 24 | How many original records this project plans to use | 0 direct; use Visual-CoT derivative: minimum 2,000, full 9,000. |
| 25 | Planned augmentations and exact quotas | Accepted targets: minimum 1,800 augmentations; full targets in Appendix C, emphasizing same-page multi-evidence and legend/header dependencies. |
| 26 | Its main job in this project — or the decision to exclude it | `exclude` from independent allocation. |
| 27 | Why that decision makes sense | The original release lacks complete spatial evidence; the derivative provides the actionable label while preserving parent provenance. |
| 28 | What still must be acquired or verified | License review and chart-element oracle recall must pass before inclusion. |

### 5.5 Visual-CoT DUDE records

> **Quick take**
> - **Main project role:** `exclude`.
> - **What it uniquely adds:** No unique contribution beyond a normalized box view.
> - **Planned original records:** 0 minimum; 0 full.
> - **Why this decision:** Original DUDE is cleaner for multi-page topology and avoids duplicate accounting.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | `dude_cot_train.jsonl`, a Visual-CoT derivative of original DUDE. |
| 2 | Where to find the paper and official release | [VCOT-GH] and [DUDE-P]. |
| 3 | Can we get it now? | Public metadata/assets subject to both releases. |
| 4 | Exact version or revision | Pinned Visual-CoT SHA in §5.1. |
| 5 | What the license allows — and does not allow | Most restrictive combination of Visual-CoT and DUDE parent terms. |
| 6 | Where the original documents came from | Original DUDE documents/questions. |
| 7 | What kind of material it contains | Multi-page documents rendered to question-associated page images. |
| 8 | How many documents | No independent document count. |
| 9 | How many pages or images | No independent page count. |
| 10 | How many questions | **What the official source says** (`primary_source_fact`) The paper rounds this view to approximately 15k rows; **What we directly verified in the release** (`verified_release_observation`) the pinned `dude_cot_train.jsonl` has exactly **6,088** lines. [VCOT-P] [VCOT-GH] |
| 11 | Official train, development, and test counts | Parent train split only should ever be considered. |
| 12 | Kinds of answers | Inherited DUDE answer types, but derivative rows emphasize answer-box matching. |
| 13 | Kinds of reasoning | Direct/extractive grounding; incomplete for abstractive/list/unanswerable records. |
| 14 | Where and how the evidence is spread | Page-image boxes rather than audited document-level evidence sets. |
| 15 | What evidence labels exist and how precise they are | Answer-matched bounding boxes. |
| 16 | How spatial coordinates are represented | Absolute pixels. |
| 17 | Are the original PDFs or page images available? | Parent assets available through DUDE/RRC. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | High for extractive rows; poor for abstraction and distributed evidence. |
| 19 | Does it label the complete evidence set? | No. |
| 20 | Known gaps or quality concerns | Duplicate boxes and incomplete evidence are especially risky for list and multi-page questions. |
| 21 | Could it overlap with other training or evaluation data? | Exact duplicate of original DUDE parent questions/documents. |
| 22 | Real example questions from official material | Use the original DUDE examples in §5.13; derivative-specific exact examples were not separately required because the records retain parent questions. |
| 23 | What this source uniquely adds | No unique contribution beyond a normalized box view. |
| 24 | How many original records this project plans to use | 0 minimum; 0 full. |
| 25 | Planned augmentations and exact quotas | 0. |
| 26 | Its main job in this project — or the decision to exclude it | `exclude`. |
| 27 | Why that decision makes sense | Original DUDE is cleaner for multi-page topology and avoids duplicate accounting. |
| 28 | What still must be acquired or verified | Keep hashes in the overlap ledger so no Visual-CoT DUDE row can enter after a source substitution. |

### 5.6 SROIE

> **Quick take**
> - **Main project role:** `exclude` as an independent QA source.
> - **What it uniquely adds:** Clean receipt-layout/field grounding for Stage 0.
> - **Planned original records:** 0 direct; Visual-CoT derivative minimum 500/full 3,000.
> - **Why this decision:** It is not a QA dataset; the useful records are already counted under Visual-CoT.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **SROIE: Scanned Receipts OCR and Information Extraction**. |
| 2 | Where to find the paper and official release | Paper: [SROIE-P]. Official competition portal: [SROIE-SITE]. |
| 3 | Can we get it now? | Available through the competition/data portal and multiple mirrors; official terms control. |
| 4 | Exact version or revision | ICDAR 2019 competition release. |
| 5 | What the license allows — and does not allow | Competition/data-use terms; mirror licenses are not treated as authoritative. |
| 6 | Where the original documents came from | Scanned receipts with line OCR and four key fields. |
| 7 | What kind of material it contains | Single receipt images; OCR/KIE, not native QA. |
| 8 | How many documents | **What the official source says** (`primary_source_fact`) 1,000 receipt images. |
| 9 | How many pages or images | 1,000 images. |
| 10 | How many questions | 0 native questions. Visual-CoT creates approximately 4k derivative QA rows. |
| 11 | Official train, development, and test counts | Paper describes 600 train/400 test; commonly used mirrors expose 626/347. Treat this as a release conflict and use official acquired manifest. |
| 12 | Kinds of answers | Native fields: company, address, date, total; derivative answers are extractive. |
| 13 | Kinds of reasoning | Key-value lookup and OCR reading. |
| 14 | Where and how the evidence is spread | One receipt, usually one field region plus context. |
| 15 | What evidence labels exist and how precise they are | Native line boxes/transcripts and key-field labels; no native questions. |
| 16 | How spatial coordinates are represented | Quadrilateral/line coordinates in task annotations; Visual-CoT derivative uses rectangular pixels. |
| 17 | Are the original PDFs or page images available? | Images and annotations available subject to competition terms. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | High for line candidates; key-value role labels can deterministically map to units. |
| 19 | Does it label the complete evidence set? | Only for the four native fields, not for generated free-form questions. |
| 20 | Known gaps or quality concerns | Split-count conflict; receipt totals and addresses can span multiple lines. |
| 21 | Could it overlap with other training or evaluation data? | Visual-CoT derivative overlaps the same images/fields; do not count native field records separately. |
| 22 | Real example questions from official material | No native question exists. Authentic Visual-CoT derivative questions include “What is the name of the restaurant?” and “What is the address of the restaurant?” [VCOT-GH] |
| 23 | What this source uniquely adds | Clean receipt-layout/field grounding for Stage 0. |
| 24 | How many original records this project plans to use | 0 direct; Visual-CoT derivative minimum 500/full 3,000. |
| 25 | Planned augmentations and exact quotas | Minimum 500 accepted derivatives; full quotas emphasize paraphrase, same-page key/value dependency, and no-evidence variants. |
| 26 | Its main job in this project — or the decision to exclude it | `exclude` as an independent QA source. |
| 27 | Why that decision makes sense | It is not a QA dataset; the useful records are already counted under Visual-CoT. |
| 28 | What still must be acquired or verified | Acquire the official manifest, reconcile 600/400 versus 626/347, and verify multi-line address evidence. |

### 5.7 TextVQA

> **Quick take**
> - **Main project role:** `exclude`.
> - **What it uniquely adds:** None for this document corpus; scene-text robustness is outside the approved source boundary.
> - **Planned original records:** 0 minimum; 0 full.
> - **Why this decision:** Scene photographs do not provide the intended document fidelity or native complete-evidence supervision; the owner has excluded both original and Visual-CoT-derived TextVQA records.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **TextVQA: Visual Question Answering with Reading Comprehension**. |
| 2 | Where to find the paper and official release | Paper: [TEXTVQA-P]. Official site/data: [TEXTVQA-SITE]. |
| 3 | Can we get it now? | Public benchmark assets and annotations. |
| 4 | Exact version or revision | Official TextVQA release used by Visual-CoT. |
| 5 | What the license allows — and does not allow | CC BY 4.0 for annotations; Open Images terms apply to images. |
| 6 | Where the original documents came from | Open Images scene photographs containing readable text. |
| 7 | What kind of material it contains | Scene-text VQA, not document pages. |
| 8 | How many documents | **What the official source says** (`primary_source_fact`) 28,408 images. |
| 9 | How many pages or images | 28,408 images. |
| 10 | How many questions | **What the official source says** (`primary_source_fact`) 45,336 questions. |
| 11 | Official train, development, and test counts | 34,602 train; 5,000 validation; 5,734 test. |
| 12 | Kinds of answers | Short answers with ten human answers/question. |
| 13 | Kinds of reasoning | Scene-text reading, entity recognition, spatial reference. |
| 14 | Where and how the evidence is spread | One scene image; usually one text region but sometimes contextual visual evidence. |
| 15 | What evidence labels exist and how precise they are | Machine OCR tokens/boxes and answer annotations; no gold evidence regions. |
| 16 | How spatial coordinates are represented | OCR pixel boxes. |
| 17 | Are the original PDFs or page images available? | Images available under Open Images terms. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Moderate; parser candidates must distinguish scene objects from document semantic units. |
| 19 | Does it label the complete evidence set? | No. |
| 20 | Known gaps or quality concerns | OCR noise and domain mismatch may teach scene-text shortcuts rather than document structure. |
| 21 | Could it overlap with other training or evaluation data? | Visual-CoT derivative uses the same questions/images. |
| 22 | Real example questions from official material | Authentic Visual-CoT source rows include “what is the name of the person?” and “what does the arrow represent for a slow time?” [VCOT-GH] |
| 23 | What this source uniquely adds | None for this document corpus; scene-text robustness is outside the approved source boundary. |
| 24 | How many original records this project plans to use | 0 minimum; 0 full. |
| 25 | Planned augmentations and exact quotas | 0. |
| 26 | Its main job in this project — or the decision to exclude it | `exclude`. |
| 27 | Why that decision makes sense | Scene photographs do not provide the intended document fidelity or native complete-evidence supervision; the owner has excluded both original and Visual-CoT-derived TextVQA records. |
| 28 | What still must be acquired or verified | Keep release identifiers in the audit/overlap ledger only; do not acquire or materialize TextVQA for this corpus. |

### 5.8 TextCaps

> **Quick take**
> - **Main project role:** `exclude`.
> - **What it uniquely adds:** No unique selector contribution to the document corpus.
> - **Planned original records:** 0 minimum; 0 full.
> - **Why this decision:** Weak document fidelity, no native questions, and redundant visuals.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **TextCaps: a Dataset for Image Captioning with Reading Comprehension**. |
| 2 | Where to find the paper and official release | Paper: [TEXTCAPS-P]. Official site/release: [TEXTCAPS-SITE]. |
| 3 | Can we get it now? | Public captioning benchmark. |
| 4 | Exact version or revision | Original release used as a parent by Visual-CoT. |
| 5 | What the license allows — and does not allow | Official release terms plus Open Images image rights. |
| 6 | Where the original documents came from | Same scene-image family as TextVQA. |
| 7 | What kind of material it contains | Image captioning with scene text; not QA and not documents. |
| 8 | How many documents | 28,408 images. |
| 9 | How many pages or images | 28,408 images. |
| 10 | How many questions | 0 native questions; 142,040 captions. Visual-CoT reports ~32k generated QA rows. |
| 11 | Official train, development, and test counts | Train/validation/test image-caption splits in the official release. |
| 12 | Kinds of answers | Native free-form captions. |
| 13 | Kinds of reasoning | Description rather than question-conditioned evidence localization. |
| 14 | Where and how the evidence is spread | One scene image. |
| 15 | What evidence labels exist and how precise they are | Captions and OCR, no question/evidence set. |
| 16 | How spatial coordinates are represented | OCR pixel boxes, not caption evidence. |
| 17 | Are the original PDFs or page images available? | Images available subject to Open Images terms. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Possible only after synthetic question generation. |
| 19 | Does it label the complete evidence set? | No. |
| 20 | Known gaps or quality concerns | Visual-CoT questions are generated from captions and risk lexical/template shortcuts. |
| 21 | Could it overlap with other training or evaluation data? | Image overlap with TextVQA and Visual-CoT TextVQA. |
| 22 | Real example questions from official material | No authentic native questions exist because TextCaps is captioning. No constructed question is presented as authentic. |
| 23 | What this source uniquely adds | No unique selector contribution to the document corpus. |
| 24 | How many original records this project plans to use | 0 minimum; 0 full. |
| 25 | Planned augmentations and exact quotas | 0. |
| 26 | Its main job in this project — or the decision to exclude it | `exclude`. |
| 27 | Why that decision makes sense | Weak document fidelity, no native questions, and redundant visuals. |
| 28 | What still must be acquired or verified | Keep release identifiers in the audit/overlap ledger only; do not acquire or materialize TextCaps for this corpus. |

### 5.9 BoundingDocs v2.0

> **Quick take**
> - **Main project role:** `auxiliary_selector_sft`.
> - **What it uniquely adds:** Large, normalized auxiliary grounding source across forms/invoices and underrepresented layouts.
> - **Planned original records:** Minimum 1,000; full 5,000, drawn only from DeepForm, FATURA, Kleister, VRDU, XFUND, and other source-distinct parents. Published source-distinct all-split ceiling after removing DUDE/MP/SP is 212,488 QA (**What we calculated from cited facts** (`derived_calculation`)).
> - **Why this decision:** Visual-CoT is the starter because its selected source-specific files are pinned and simpler to audit. BoundingDocs is capped as auxiliary expansion because its heterogeneous unified mixture requires parent-license resolution, removal of DUDE/MP/SP overlap, reproduction of v2 alignment fixes, and source-balanced sampling. This does not claim stronger evidence semantics for Visual-CoT; both remain Level C until complete-evidence verification.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **BoundingDocs: A Unified Dataset for Document Question Answering with Bounding Boxes**. |
| 2 | Where to find the paper and official release | Paper: [BOUND-P]. Official HF release: [BOUND-HF]. |
| 3 | Can we get it now? | Public HF dataset. |
| 4 | Exact version or revision | **What we directly verified in the release** (`verified_release_observation`) v2.0; v2 corrects MP-DocVQA image/answer alignment. |
| 5 | What the license allows — and does not allow | CC BY 4.0 at the dataset-card level, subject to every parent source license. |
| 6 | Where the original documents came from | Unifies DeepForm, DUDE, FATURA, FUNSD, Kleister Charity/NDA, MP-/SP-DocVQA, VRDU Ad/Registration, and XFUND. |
| 7 | What kind of material it contains | Single- and multi-page business documents normalized to a common QA/box schema. |
| 8 | How many documents | 48,151 documents: 38,515 train; 4,818 validation; 4,818 test. |
| 9 | How many pages or images | 237,437 pages. |
| 10 | How many questions | 249,016 QA. Per-source totals are listed in Appendix A. |
| 11 | Official train, development, and test counts | Document split counts above; source-specific question splits are inherited/normalized. |
| 12 | Kinds of answers | Predominantly extractive/key-field and document QA answers. |
| 13 | Kinds of reasoning | Lookup, field extraction, form/table reading; limited native multi-evidence dependency. |
| 14 | Where and how the evidence is spread | One or more answer boxes, potentially multi-page through parent datasets. |
| 15 | What evidence labels exist and how precise they are | Answer boxes unified from parent annotations or matching procedures. |
| 16 | How spatial coordinates are represented | OCR boxes normalized to `[0,1]`; answer boxes normalized to `[0,1000]` in v2. |
| 17 | Are the original PDFs or page images available? | Rendered images/assets provided through HF subject to parent rights. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | High for source-distinct extractive records; boxes snap naturally to OCR/semantic units. |
| 19 | Does it label the complete evidence set? | No general completeness guarantee. |
| 20 | Known gaps or quality concerns | Parent unification obscures heterogeneous annotation quality. Answer boxes may omit headers/qualifiers. |
| 21 | Could it overlap with other training or evaluation data? | Direct overlap with DUDE, MP-DocVQA, SP-DocVQA/DocVQA, FUNSD, and other public corpora. Source-distinct eligible pool excludes DUDE/MP/SP. |
| 22 | Real example questions from official material | Paper examples include “What is the total due after the current charges?” and “What name is written on the invoice?” [BOUND-P] |
| 23 | What this source uniquely adds | Large, normalized auxiliary grounding source across forms/invoices and underrepresented layouts. |
| 24 | How many original records this project plans to use | Minimum 1,000; full 5,000, drawn only from DeepForm, FATURA, Kleister, VRDU, XFUND, and other source-distinct parents. Published source-distinct all-split ceiling after removing DUDE/MP/SP is 212,488 QA (**What we calculated from cited facts** (`derived_calculation`)). |
| 25 | Planned augmentations and exact quotas | Minimum 600 accepted augmentations; full per-family allocations in Appendix C, with source caps and strict parent provenance. |
| 26 | Its main job in this project — or the decision to exclude it | `auxiliary_selector_sft`. |
| 27 | Why that decision makes sense | Visual-CoT is the starter because its selected source-specific files are pinned and simpler to audit. BoundingDocs is capped as auxiliary expansion because its heterogeneous unified mixture requires parent-license resolution, removal of DUDE/MP/SP overlap, reproduction of v2 alignment fixes, and source-balanced sampling. This does not claim stronger evidence semantics for Visual-CoT; both remain Level C until complete-evidence verification. |
| 28 | What still must be acquired or verified | Acquire source-level manifests, enforce parent licenses, and independently reproduce v2 alignment fixes before sampling. |

### 5.10 BBox DocVQA v1 and SciEGQA v2

> **Quick take**
> - **Main project role:** `sealed_evaluation` for SciEGQA-Bench; v1/train are `exclude` pending license.
> - **What it uniquely adds:** Potential high-value scientific multi-region/multi-page evidence source after legal and quality gates.
> - **Planned original records:** 0 minimum; 0 full at protocol v1. A future licensed addendum may substitute up to 4,000 records without changing totals.
> - **Why this decision:** Prevents double-counting v1/v2 and avoids training on an unlicensed automatic release.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **BBox DocVQA** is arXiv:2511.15090v1. The same arXiv record was renamed/redefined in v2 as **SciEGQA: Scientific Evidence Grounded Question Answering**. |
| 2 | Where to find the paper and official release | Historic v1: [BBOX-V1]. Current paper/project/release: [SCIEGQA-P], [SCIEGQA-PROJ], [SCIEGQA-GH], [SCIEGQA-TRAIN], [SCIEGQA-BENCH]. |
| 3 | Can we get it now? | SciEGQA train and benchmark repositories are public; v1 is superseded and not a separately released current dataset. |
| 4 | Exact version or revision | v1 BBox DocVQA historical; SciEGQA arXiv v2 dated 2026-03-30. HF revisions must be recorded at acquisition. |
| 5 | What the license allows — and does not allow | **Still unresolved** (`unresolved`) No explicit dataset license was located in the official SciEGQA repositories/cards at the cutoff. |
| 6 | Where the original documents came from | Academic-paper pages and automatically generated/verified scientific evidence questions. |
| 7 | What kind of material it contains | Scientific paper page images with one- or two-page evidence boxes. |
| 8 | How many documents | Release is row-based; distinct paper/document counts must be derived from acquired metadata. |
| 9 | How many pages or images | SciEGQA-Train repository size 77.1 GB; SciEGQA-Bench 1.26 GB. Exact unique page count requires the manifest. |
| 10 | How many questions | 30,780 train rows; 1,623 benchmark rows. |
| 11 | Official train, development, and test counts | Train and benchmark are separate repositories; no benchmark row may enter development. |
| 12 | Kinds of answers | Extractive and reasoning answers grounded to scientific evidence. |
| 13 | Kinds of reasoning | Direct, multi-region, and multi-page scientific evidence reasoning. |
| 14 | Where and how the evidence is spread | Benchmark: 1,041 single-page/single-region; 186 single-page/multi-region; 396 multi-page/multi-region. |
| 15 | What evidence labels exist and how precise they are | `evidence_page` plus absolute `bbox` and `rel_bbox`; train is automatically generated, benchmark human-verified. |
| 16 | How spatial coordinates are represented | Absolute image pixels and relative normalized boxes. |
| 17 | Are the original PDFs or page images available? | Page images are packaged in HF repositories. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | High if boxes correspond to semantic candidates; strict oracle coverage is measurable. |
| 19 | Does it label the complete evidence set? | Benchmark aims at grounded evidence; train completeness remains a pseudo-label claim requiring independent audit. |
| 20 | Known gaps or quality concerns | Automatic train generation may encode teacher shortcuts; exact parent-paper provenance and duplicates require checking. |
| 21 | Could it overlap with other training or evaluation data? | Potential overlap with arXiv/PaperQA/SPIQA/QASPER scientific papers and any future MMLongBench sources. |
| 22 | Real example questions from official material | Official project figures contain examples, but two exact question strings were not recoverable from the accessible text layer. No constructed examples are substituted. |
| 23 | What this source uniquely adds | Potential high-value scientific multi-region/multi-page evidence source after legal and quality gates. |
| 24 | How many original records this project plans to use | 0 minimum; 0 full at protocol v1. A future licensed addendum may substitute up to 4,000 records without changing totals. |
| 25 | Planned augmentations and exact quotas | 0 in the present plan. Any addendum requires native-only, paraphrase, same-page, and real-cross-page strata with the same verifier. |
| 26 | Its main job in this project — or the decision to exclude it | `sealed_evaluation` for SciEGQA-Bench; v1/train are `exclude` pending license. |
| 27 | Why that decision makes sense | Prevents double-counting v1/v2 and avoids training on an unlicensed automatic release. |
| 28 | What still must be acquired or verified | Obtain written license/terms, freeze HF revision, hash source PDFs, reproduce category counts, audit ≥200 train rows, and quarantine all benchmark assets. |

### 5.11 GroundingDocQA / M3Grounder data

> **Quick take**
> - **Main project role:** `exclude`.
> - **What it uniquely adds:** Would fill native mask-grounded multi-span supervision if released.
> - **Planned original records:** 0 minimum; 0 full.
> - **Why this decision:** No acquisition path, license, or auditable release.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **GroundingDocQA**, associated with **M3Grounder: Mask-Based Multi-Span and Multi-Granular Grounding for Document QA**. |
| 2 | Where to find the paper and official release | Project: [M3G-PROJ]. Paper: [M3G-P]. |
| 3 | Can we get it now? | **What we directly verified in the release** (`verified_release_observation`) Not publicly downloadable as of 2026-08-06; project page says “Dataset Soon.” |
| 4 | Exact version or revision | CVPR 2026 paper/project snapshot. |
| 5 | What the license allows — and does not allow | **Still unresolved** (`unresolved`) No usable dataset license or terms because assets are not released. |
| 6 | Where the original documents came from | Large-scale document collection assembled for phrase/line/block mask grounding. |
| 7 | What kind of material it contains | Document images with multi-granular segmentation masks. |
| 8 | How many documents | Paper/project report approximately 200k documents. |
| 9 | How many pages or images | Not independently verifiable without release. |
| 10 | How many questions | Project reports approximately 2M QA; paper describes a 732k training subset and 368-question benchmark over 160 images. |
| 11 | Official train, development, and test counts | No downloadable split manifest. |
| 12 | Kinds of answers | Document QA with grounded spans/masks. |
| 13 | Kinds of reasoning | Multi-span and multi-granular grounding. |
| 14 | Where and how the evidence is spread | One or multiple phrase/line/block masks, potentially distributed. |
| 15 | What evidence labels exist and how precise they are | Pixel masks at phrase, line, and block granularity. |
| 16 | How spatial coordinates are represented | Pixel masks. |
| 17 | Are the original PDFs or page images available? | No public assets at cutoff. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Expected high, but untestable. |
| 19 | Does it label the complete evidence set? | Claimed stronger than answer boxes; completeness cannot be independently assessed. |
| 20 | Known gaps or quality concerns | Unreleased data prevents schema, duplicate, annotation-quality, and licensing audit. |
| 21 | Could it overlap with other training or evaluation data? | Unknown overlap with DUDE/DocVQA/other synthetic sources. |
| 22 | Real example questions from official material | Official project images show examples, but exact question strings were not available in accessible text. No constructed examples are substituted. |
| 23 | What this source uniquely adds | Would fill native mask-grounded multi-span supervision if released. |
| 24 | How many original records this project plans to use | 0 minimum; 0 full. |
| 25 | Planned augmentations and exact quotas | 0. |
| 26 | Its main job in this project — or the decision to exclude it | `exclude`. |
| 27 | Why that decision makes sense | No acquisition path, license, or auditable release. |
| 28 | What still must be acquired or verified | Re-audit only after public assets, schema, split manifest, license, and stable revision exist. |

### 5.12 MP-DocVQA

> **Quick take**
> - **Main project role:** `calibration_or_development`.
> - **What it uniquely adds:** Calibrates wrong-page negatives and page-gate recall.
> - **Planned original records:** 0 selector-SFT records; development/calibration only.
> - **Why this decision:** Valuable for page retrieval but insufficient as complete region-evidence SFT.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **MP-DocVQA: Modeling Long Multi-Page Documents for Visual Question Answering**. |
| 2 | Where to find the paper and official release | Paper: [MPDOC-P]. Official framework/release: [MPDOC-GH]. Dataset card/mirror used only to document count conflict: [MPDOC-HF]. |
| 3 | Can we get it now? | Available through official/RRC channels and framework instructions. |
| 4 | Exact version or revision | Original release; exact acquired manifest must be pinned. |
| 5 | What the license allows — and does not allow | RRC/data-use terms; repository code license does not automatically license document assets. |
| 6 | Where the original documents came from | Multi-page documents derived from the same industry-document ecosystem as DocVQA. |
| 7 | What kind of material it contains | Multi-page scanned business documents with page-level answer localization. |
| 8 | How many documents | 6,000 documents. |
| 9 | How many pages or images | Approximately 49,000 pages (`derived` from reported mean 8.27 pages/document); use manifest for exact count. |
| 10 | How many questions | **What the official source says** (`primary_source_fact`) The MP-DocVQA paper reports **46,176 questions over 6,000 documents**. The current HF mirror is used only for availability/schema; it is not treated as an independent count authority. [MPDOC-P] [MPDOC-HF] |
| 11 | Official train, development, and test counts | Official train/validation/test partitions are document-disjoint; exact row counts must be read from acquired release. |
| 12 | Kinds of answers | Short mostly extractive answers. |
| 13 | Kinds of reasoning | Page retrieval plus answer extraction; most questions are answerable on one page. |
| 14 | Where and how the evidence is spread | Multiple pages with one gold answer page and distractor pages; not genuine cross-page evidence in most records. |
| 15 | What evidence labels exist and how precise they are | Question/answer, OCR, and answer-page index; no complete multi-region set. |
| 16 | How spatial coordinates are represented | Page/OCR coordinates inherited from release. |
| 17 | Are the original PDFs or page images available? | Document pages available under RRC terms. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Useful for page candidate/oracle calibration, less useful for region labels. |
| 19 | Does it label the complete evidence set? | No. |
| 20 | Known gaps or quality concerns | Multi-page container can be mistaken for multi-hop reasoning; page labels do not identify all supporting regions. |
| 21 | Could it overlap with other training or evaluation data? | Overlaps DocVQA family, BoundingDocs MP-DocVQA, and MMDocIR training mixture. |
| 22 | Real example questions from official material | Official paper examples include page-retrieval questions, but two exact strings were not reliably recoverable from the accessible text layer; no constructed examples are substituted. |
| 23 | What this source uniquely adds | Calibrates wrong-page negatives and page-gate recall. |
| 24 | How many original records this project plans to use | 0 selector-SFT records; development/calibration only. |
| 25 | Planned augmentations and exact quotas | No generative augmentation; ordinary wrong-page negatives may be mined for calibration, never promoted to gold without checks. |
| 26 | Its main job in this project — or the decision to exclude it | `calibration_or_development`. |
| 27 | Why that decision makes sense | Valuable for page retrieval but insufficient as complete region-evidence SFT. |
| 28 | What still must be acquired or verified | Obtain the official split manifest during acquisition, reproduce the paper total of 46,176, and hash documents against DocVQA, BoundingDocs, and MMDocIR. |

### 5.13 DUDE — original release

> **Quick take**
> - **Main project role:** `core_selector_sft`.
> - **What it uniquely adds:** Primary real-document source for multi-page, list, and native unanswerable behavior.
> - **Planned original records:** Minimum 4,500; full 12,500 from train only, with extractive completeness audit and stratification across answer types.
> - **Why this decision:** It uniquely supplies real multi-page documents and native unanswerables at useful scale.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **DUDE: Document Understanding Dataset and Evaluation / Document UnderstanDing of Everything**. |
| 2 | Where to find the paper and official release | Paper: [DUDE-P]. Official repository/portal: [DUDE-GH]. Official loader/card: [DUDE-HF]. |
| 3 | Can we get it now? | Public with RRC registration; PDFs, OCR, and annotations are available. |
| 4 | Exact version or revision | ICDAR/ICCV 2023 release; repository snapshot SHA listed in source ledger. |
| 5 | What the license allows — and does not allow | Dataset/loader reported CC BY 4.0; repository tools GPL-3.0. Treat document binaries under the dataset terms, not GPL. |
| 6 | Where the original documents came from | Diverse real multi-page PDFs from multiple document domains. |
| 7 | What kind of material it contains | Multi-page PDF documents with OCR from Azure, Amazon Textract, and Tesseract. |
| 8 | How many documents | Conflict: abstract 5,019 documents; Table 2 sums to 4,974. Train 3,010; validation 749; test 1,215. |
| 9 | How many pages or images | Abstract 23,707 pages; Table 2 split sum 23,707 (14,271/3,697/5,739). |
| 10 | How many questions | Conflict: abstract 41,541; Table 2 split sum 41,491 (23,728/6,315/11,448). |
| 11 | Official train, development, and test counts | Train 23,728; validation 6,315; test 11,448 questions per Table 2. |
| 12 | Kinds of answers | Extractive, abstractive, list, and unanswerable. |
| 13 | Kinds of reasoning | Direct lookup, list aggregation, document-wide search, temporal/financial lookup, abstraction. |
| 14 | Where and how the evidence is spread | Zero evidence for unanswerable; one/multiple regions; multiple pages; global/document scope. |
| 15 | What evidence labels exist and how precise they are | Native evidence boxes for extractive answers; non-extractive categories lack equivalent complete spatial labels. |
| 16 | How spatial coordinates are represented | OCR-specific coordinates and PDF/page identifiers. |
| 17 | Are the original PDFs or page images available? | Original PDFs, renderings, and OCR available through RRC. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | High for extractive boxes; Level B/C alignment for list/abstractive requires new evidence annotation. |
| 19 | Does it label the complete evidence set? | Only reliably for audited extractive rows. Unanswerable is native; list/abstractive evidence is incomplete. |
| 20 | Known gaps or quality concerns | Published total conflict; OCR providers differ; some annotations may identify an answer span rather than all explanatory context. |
| 21 | Could it overlap with other training or evaluation data? | Overlaps Visual-CoT DUDE, BoundingDocs DUDE, and MMDocIR training mixture. |
| 22 | Real example questions from official material | Paper examples include “What ratio was used for Payout to Reported Household Income?” and “What was the total of securities transactions settlements including interest?” [DUDE-P] |
| 23 | What this source uniquely adds | Primary real-document source for multi-page, list, and native unanswerable behavior. |
| 24 | How many original records this project plans to use | Minimum 4,500; full 12,500 from train only, with extractive completeness audit and stratification across answer types. |
| 25 | Planned augmentations and exact quotas | Minimum 4,200 accepted augmentations; full matrix emphasizes 4,000 real cross-page, 1,800 semi-extractive, 1,600 abstractive, missing-premise, and query negatives. |
| 26 | Its main job in this project — or the decision to exclude it | `core_selector_sft`. |
| 27 | Why that decision makes sense | It uniquely supplies real multi-page documents and native unanswerables at useful scale. |
| 28 | What still must be acquired or verified | Acquire manifest and resolve published count conflict; independently inspect list/abstractive evidence; hash against all derivative mixtures. |


## 6. Datasets with synthesis, reasoning, or multiple pieces of evidence

**Friendly guidepost.** These sources bring harder questions: explanations, calculations, comparisons, conditions, and evidence spread across paragraphs, tables, pages, or documents. Many are excellent for reasoning, but only some are ready to supervise a spatial evidence selector. The 28-row cards use the same reading order as Section 5, so you can compare sources field by field.

These sources are separated from native spatial data. Paragraphs, facts, pages, and programs are Level B only after deterministic spatial alignment; answer-only records remain Level D.

### 6.1 SlideVQA

> **Quick take**
> - **Main project role:** `sealed_evaluation`.
> - **What it uniquely adds:** High-value external test of real cross-slide selection and arithmetic.
> - **Planned original records:** 0 minimum; 0 full.
> - **Why this decision:** Excellent task fit but explicit license restriction.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **SlideVQA: A Dataset for Document Visual Question Answering on Multiple Images**. |
| 2 | Where to find the paper and official release | Paper: [SLIDE-P]. Official repository: [SLIDE-GH]. License: [SLIDE-LIC]. |
| 3 | Can we get it now? | Annotations and assets are publicly visible, but use is restricted. |
| 4 | Exact version or revision | Official repository main snapshot and paper v1. |
| 5 | What the license allows — and does not allow | **What we directly verified in the release** (`verified_release_observation`) Evaluation/internal use only; training and derivative-data generation are not authorized by the repository license. |
| 6 | Where the original documents came from | Real slide decks collected from the web. |
| 7 | What kind of material it contains | Multi-image slide decks with text, figures, charts, and layout. |
| 8 | How many documents | 2,619 decks: train 2,023; dev 246; test 350. |
| 9 | How many pages or images | 52,586 slides: train 40,340; dev 5,663; test 6,583. |
| 10 | How many questions | 14,484 questions: train 8,506; dev 2,686; test 3,292. |
| 11 | Official train, development, and test counts | As above; official train is not used because the license bars training. |
| 12 | Kinds of answers | Extractive, numerical, comparison, and deck-level answers. |
| 13 | Kinds of reasoning | Retrieval, arithmetic, comparison, aggregation, cross-slide lookup. |
| 14 | Where and how the evidence is spread | Multiple evidence slides in a real deck; paper reports mean 5.91 evidence slides/question. |
| 15 | What evidence labels exist and how precise they are | `answer_page_idx`, `answer_node`, and bounding-box annotations at slide-object level. |
| 16 | How spatial coordinates are represented | Repository bounding boxes and slide dimensions; normalize at ingestion. |
| 17 | Are the original PDFs or page images available? | Slide images and annotations exist. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | High for page/object candidates. |
| 19 | Does it label the complete evidence set? | Page/object labels are stronger than answer-only but do not necessarily prove every labeled slide is necessary. |
| 20 | Known gaps or quality concerns | License precludes training; evidence-slide sets may be generous rather than minimal. |
| 21 | Could it overlap with other training or evaluation data? | Possible web-slide overlap with LongDocURL/MMDocIR; quarantine hashes. |
| 22 | Real example questions from official material | Official examples: “How many acquisitions did IBM make?” and “In what year did IBM acquire Cognos?” [SLIDE-P] |
| 23 | What this source uniquely adds | High-value external test of real cross-slide selection and arithmetic. |
| 24 | How many original records this project plans to use | 0 minimum; 0 full. |
| 25 | Planned augmentations and exact quotas | 0; no augmentation seeds. |
| 26 | Its main job in this project — or the decision to exclude it | `sealed_evaluation`. |
| 27 | Why that decision makes sense | Excellent task fit but explicit license restriction. |
| 28 | What still must be acquired or verified | Freeze repo revision and obtain owner permission before any future training use; until then keep all questions/prompts inaccessible to developers. |

### 6.2 QASPER

> **Quick take**
> - **Main project role:** `auxiliary_selector_sft`.
> - **What it uniquely adds:** Realistic user-authored scientific questions with paragraph evidence and unanswerables.
> - **Planned original records:** Minimum 1,000; full 1,100 aligned train questions.
> - **Why this decision:** High realism and evidence quality outweigh limited scale; PDF alignment upgrades it to Level B/C.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **QASPER: A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers**. |
| 2 | Where to find the paper and official release | Paper: [QASPER-P]. Official HF release: [QASPER-HF]. |
| 3 | Can we get it now? | Public text/structure/evidence annotations; source papers must be reacquired under their own rights. |
| 4 | Exact version or revision | HF release observed 2022-10-07; paper NAACL 2021. |
| 5 | What the license allows — and does not allow | CC BY 4.0 for dataset annotations. |
| 6 | Where the original documents came from | 1,585 NLP research papers from S2ORC; questions written by paper readers and answered by NLP practitioners. |
| 7 | What kind of material it contains | Long scientific papers represented as section/paragraph text; PDFs can be reacquired where lawful. |
| 8 | How many documents | 1,585 papers: train 888; dev 281; test 416. |
| 9 | How many pages or images | Page count not published because release is structured text; derive only after PDF acquisition/rendering. |
| 10 | How many questions | 5,049 questions and 8,207 answer annotations. |
| 11 | Official train, development, and test counts | Train 2,593; dev 1,005; test 1,451 questions. |
| 12 | Kinds of answers | Extractive, abstractive, yes/no, and unanswerable. |
| 13 | Kinds of reasoning | Information seeking, comparison, methods/results, explanation, limitations. |
| 14 | Where and how the evidence is spread | One or multiple supporting paragraphs; potentially multiple pages after rendering. |
| 15 | What evidence labels exist and how precise they are | Native evidence paragraph IDs for each answer annotation. |
| 16 | How spatial coordinates are represented | No native boxes; derived PDF/page/pixel coordinates after alignment. |
| 17 | Are the original PDFs or page images available? | Original PDFs are not universally packaged; acquisition must preserve paper IDs and licenses. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | High after exact/semantic paragraph alignment; figures/tables referenced only indirectly need special handling. |
| 19 | Does it label the complete evidence set? | Often strong textual evidence, but alternative supporting paragraphs and figure dependencies require verification. |
| 20 | Known gaps or quality concerns | Multiple annotator answers/evidence can disagree; structured text may omit visual information. |
| 21 | Could it overlap with other training or evaluation data? | Potential paper overlap with SciEGQA, SPIQA, PeerQA, and arXiv-derived benchmarks. |
| 22 | Real example questions from official material | Official examples include “What datasets are used to evaluate?” and “Is the model computed efficiently?” [QASPER-P] |
| 23 | What this source uniquely adds | Realistic user-authored scientific questions with paragraph evidence and unanswerables. |
| 24 | How many original records this project plans to use | Minimum 1,000; full 1,100 aligned train questions. |
| 25 | Planned augmentations and exact quotas | Minimum 700 accepted derivatives; full: 500 paraphrases, 400 compositions, 1,500 real cross-page, 500 synthetic bundles, 1,300 semi-extractive, 1,200 abstractive, 400 missing-premise, 400 query negatives. |
| 26 | Its main job in this project — or the decision to exclude it | `auxiliary_selector_sft`. |
| 27 | Why that decision makes sense | High realism and evidence quality outweigh limited scale; PDF alignment upgrades it to Level B/C. |
| 28 | What still must be acquired or verified | Acquire paper PDFs legally, verify section/paragraph hashes, reconcile multiple answer annotations, and reserve dev/test untouched. |

### 6.3 FinQA

> **Quick take**
> - **Main project role:** `planner_or_decomposition_pretraining`.
> - **What it uniquely adds:** Clean reasoning-program supervision for planner/decomposition pretraining.
> - **Planned original records:** Selector: 0. Planner pool: 6,000 train records.
> - **Why this decision:** Programs are valuable; lack of native page images/coordinates prevents direct selector SFT.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **FinQA: A Dataset of Numerical Reasoning over Financial Data**. |
| 2 | Where to find the paper and official release | Paper: [FINQA-P]. Official repository: [FINQA-GH]. |
| 3 | Can we get it now? | Public JSON annotations and code. |
| 4 | Exact version or revision | Repository main snapshot; train/dev/test file SHAs recorded in source ledger. |
| 5 | What the license allows — and does not allow | Repository software is MIT; **Still unresolved** (`unresolved`) the data/document redistribution rights must be read separately from code. |
| 6 | Where the original documents came from | Financial reports with curated tables and text. |
| 7 | What kind of material it contains | Structured text/table contexts; not a native page-image corpus. |
| 8 | How many documents | 2,789 reports. |
| 9 | How many pages or images | Not published as a visual page count. |
| 10 | How many questions | 8,281 questions. |
| 11 | Official train, development, and test counts | 6,251 train; 883 dev; 1,147 test. |
| 12 | Kinds of answers | Numerical/program-derived answers. |
| 13 | Kinds of reasoning | Addition, subtraction, multiplication, division, ratios, percentages, comparison. |
| 14 | Where and how the evidence is spread | Multiple supporting facts in a curated context; typically one source report. |
| 15 | What evidence labels exist and how precise they are | Native supporting facts and executable reasoning programs. |
| 16 | How spatial coordinates are represented | No spatial coordinates. |
| 17 | Are the original PDFs or page images available? | Context JSON is available; original report pages require reacquisition. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Operands and facts can align to candidates after rendering, but spatial fidelity is not native. |
| 19 | Does it label the complete evidence set? | Complete at the fact/program level more often than at the document-region level. |
| 20 | Known gaps or quality concerns | Curated context removes real distractors; report/page mapping may be incomplete. |
| 21 | Could it overlap with other training or evaluation data? | 2,119 MultiHiertt questions are copied from FinQA; remove overlap before combined use. DocFinQA also inherits FinQA. |
| 22 | Real example questions from official material | Official examples include “What was the percentage change in the research and development expenses from 2018 to 2019?” and “What was the average employee cost during FY18 and FY19?” [FINQA-P] |
| 23 | What this source uniquely adds | Clean reasoning-program supervision for planner/decomposition pretraining. |
| 24 | How many original records this project plans to use | Selector: 0. Planner pool: 6,000 train records. |
| 25 | Planned augmentations and exact quotas | No selector augmentations. Planner may derive subgoals/program traces; those are never treated as boxes without independent alignment. |
| 26 | Its main job in this project — or the decision to exclude it | `planner_or_decomposition_pretraining`. |
| 27 | Why that decision makes sense | Programs are valuable; lack of native page images/coordinates prevents direct selector SFT. |
| 28 | What still must be acquired or verified | Clarify data rights, remove MultiHiertt overlap, and reacquire reports only if a future visual-alignment addendum is approved. |

### 6.4 TAT-QA

> **Quick take**
> - **Main project role:** `planner_or_decomposition_pretraining`.
> - **What it uniquely adds:** Program/operand and table-text decomposition supervision.
> - **Planned original records:** Selector: 0. Planner pool: 8,000; answerer pool: 3,000 disjoint records.
> - **Why this decision:** TAT-DQA is the fair visual source; TAT-QA remains useful for structured reasoning.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance**. |
| 2 | Where to find the paper and official release | Paper: [TATQA-P]. Official repository/project: [TATQA-GH]. |
| 3 | Can we get it now? | Public JSON data and code. |
| 4 | Exact version or revision | Repository latest observed commit `870accc41953dcde885aabeb963d94aabdc0fbc3`. |
| 5 | What the license allows — and does not allow | MIT repository license; underlying annual-report rights still apply. |
| 6 | Where the original documents came from | 182 financial reports with manually selected tables and relevant paragraphs. |
| 7 | What kind of material it contains | Curated table-plus-text contexts, not original page images. |
| 8 | How many documents | 182 reports; 2,757 hybrid contexts. |
| 9 | How many pages or images | No native page count in the QA release. |
| 10 | How many questions | 16,552 questions. |
| 11 | Official train, development, and test counts | 13,215 train; 1,668 dev; 1,669 test. |
| 12 | Kinds of answers | Span, multi-span, count, arithmetic. |
| 13 | Kinds of reasoning | Table-text retrieval, aggregation, percentage, comparison, counting. |
| 14 | Where and how the evidence is spread | Multiple table cells/paragraphs within a curated context. |
| 15 | What evidence labels exist and how precise they are | Derivation, answer source, relevant paragraphs, and table cells/facts. |
| 16 | How spatial coordinates are represented | No native page coordinates. |
| 17 | Are the original PDFs or page images available? | Contexts available; TAT-DQA restores page images for overlapping questions. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Good semantic alignment, but use TAT-DQA for visual selector data. |
| 19 | Does it label the complete evidence set? | Strong at fact/program level; headers and visual structure require TAT-DQA alignment. |
| 20 | Known gaps or quality concerns | Curated contexts remove document distractors and can make retrieval unrealistically easy. |
| 21 | Could it overlap with other training or evaluation data? | Parent of TAT-DQA; do not count overlapping questions in both selector and planner/answerer pools unless objective-specific IDs are disjoint. |
| 22 | Real example questions from official material | Official examples include “How many years show an effective tax rate of 30% or higher?” and “What is the average effective tax rate from 2012 to 2018?” [TATQA-P] |
| 23 | What this source uniquely adds | Program/operand and table-text decomposition supervision. |
| 24 | How many original records this project plans to use | Selector: 0. Planner pool: 8,000; answerer pool: 3,000 disjoint records. |
| 25 | Planned augmentations and exact quotas | Planner subgoal/program traces only; visual derivatives use TAT-DQA and retain parent IDs. |
| 26 | Its main job in this project — or the decision to exclude it | `planner_or_decomposition_pretraining`. |
| 27 | Why that decision makes sense | TAT-DQA is the fair visual source; TAT-QA remains useful for structured reasoning. |
| 28 | What still must be acquired or verified | Create an explicit TAT-QA↔TAT-DQA mapping and enforce objective-disjoint record IDs. |

### 6.5 MultiHiertt

> **Quick take**
> - **Main project role:** `planner_or_decomposition_pretraining`.
> - **What it uniquely adds:** Multi-table program and evidence-set planning.
> - **Planned original records:** Selector: 0. Planner pool: 5,500 overlap-clean train questions.
> - **Why this decision:** Fills multi-table reasoning without pretending structured contexts are real pages.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **MultiHiertt: Numerical Reasoning over Multi-Hierarchical Tables**. |
| 2 | Where to find the paper and official release | Paper: [MULTIHIERTT-P]. Official repository: [MULTIHIERTT-GH]. |
| 3 | Can we get it now? | Public annotations/code. |
| 4 | Exact version or revision | Repository latest observed commit `45bd9ccdf3142ea059bd5e69c0afb83437fa539c`. |
| 5 | What the license allows — and does not allow | Apache-2.0 repository license; source-report rights remain relevant. |
| 6 | Where the original documents came from | Financial reports and hierarchical tables; 2,119 questions are inherited from FinQA. |
| 7 | What kind of material it contains | Structured multi-table plus text contexts. |
| 8 | How many documents | 2,513 reports. |
| 9 | How many pages or images | No native page-image count. |
| 10 | How many questions | 10,440 questions. |
| 11 | Official train, development, and test counts | 7,830 train; 1,044 dev; 1,566 test. |
| 12 | Kinds of answers | Numerical/program-derived answers. |
| 13 | Kinds of reasoning | Multi-table reasoning, table-text composition, aggregation, comparison, percentages. |
| 14 | Where and how the evidence is spread | Multiple cells/tables/paragraphs; often distributed within a report context. |
| 15 | What evidence labels exist and how precise they are | Supporting facts and executable programs. |
| 16 | How spatial coordinates are represented | No spatial coordinates. |
| 17 | Are the original PDFs or page images available? | Structured contexts available; original pages require reacquisition. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Possible after rendering, but not native and table geometry may be lost. |
| 19 | Does it label the complete evidence set? | Strong fact/program completeness; spatial completeness unresolved. |
| 20 | Known gaps or quality concerns | Automatic/semi-automatic context extraction can omit headers or report structure. |
| 21 | Could it overlap with other training or evaluation data? | Remove the 2,119 FinQA-overlap questions before using both datasets. |
| 22 | Real example questions from official material | Official paper examples include “What portion of total identifiable net assets is in cash?” and “Which segment had the most funds in 2017?” [MULTIHIERTT-P] |
| 23 | What this source uniquely adds | Multi-table program and evidence-set planning. |
| 24 | How many original records this project plans to use | Selector: 0. Planner pool: 5,500 overlap-clean train questions. |
| 25 | Planned augmentations and exact quotas | No selector augmentation; programs may seed verified question composition. |
| 26 | Its main job in this project — or the decision to exclude it | `planner_or_decomposition_pretraining`. |
| 27 | Why that decision makes sense | Fills multi-table reasoning without pretending structured contexts are real pages. |
| 28 | What still must be acquired or verified | Reproduce the 2,119-question overlap list and verify source-report rights. |

### 6.6 ConditionalQA

> **Quick take**
> - **Main project role:** `planner_or_decomposition_pretraining`.
> - **What it uniquely adds:** Condition/exception and missing-premise planning; Stage-5 design anchor.
> - **Planned original records:** Selector: 0. Planner pool: 2,000 train records.
> - **Why this decision:** Valuable reasoning semantics but no native visual grounding.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **ConditionalQA: A Complex Reading Comprehension Dataset with Conditional Answers**. |
| 2 | Where to find the paper and official release | Paper: [COND-P]. Official repository: [COND-GH]. |
| 3 | Can we get it now? | Public text annotations. |
| 4 | Exact version or revision | v1.0 repository release. |
| 5 | What the license allows — and does not allow | CC BY-NC-SA 4.0. |
| 6 | Where the original documents came from | UK government policy pages and scenario-conditioned questions. |
| 7 | What kind of material it contains | Long policy text; not native document images. |
| 8 | How many documents | Each record links to a policy document/page; exact unique-document count must be computed from release. |
| 9 | How many pages or images | No native page-image count. |
| 10 | How many questions | 3,427 total: 3,102 original plus 325 augmented scenario variants. |
| 11 | Official train, development, and test counts | 2,338 train; 285 dev; 804 test. |
| 12 | Kinds of answers | Yes/no, span, multi-span, and not-answerable. |
| 13 | Kinds of reasoning | Condition plus exception, scenario application, missing premise. |
| 14 | Where and how the evidence is spread | One or multiple evidence spans; zero evidence for unanswerable/insufficient cases. |
| 15 | What evidence labels exist and how precise they are | Native evidence spans and conditions. |
| 16 | How spatial coordinates are represented | Character/text spans; no pixels. |
| 17 | Are the original PDFs or page images available? | HTML/text contexts available; visual rendering would be synthetic. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Deterministic text-to-candidate alignment possible in rendered policy pages. |
| 19 | Does it label the complete evidence set? | Often strong for policy conditions, but alternate passages/implicit conditions need audit. |
| 20 | Known gaps or quality concerns | Scenario variants can be near-duplicates; unanswerable must be distinguished from omitted annotation. |
| 21 | Could it overlap with other training or evaluation data? | Template/entity leakage across variants; group by source page and scenario family. |
| 22 | Real example questions from official material | Official release examples include “Can I get Housing Benefit to pay my second home?” and “Can I claim universal credit if I get child tax credit?” [COND-GH] |
| 23 | What this source uniquely adds | Condition/exception and missing-premise planning; Stage-5 design anchor. |
| 24 | How many original records this project plans to use | Selector: 0. Planner pool: 2,000 train records. |
| 25 | Planned augmentations and exact quotas | No direct selector augmentation; selected policies may seed verified missing-premise constructions after rendering. |
| 26 | Its main job in this project — or the decision to exclude it | `planner_or_decomposition_pretraining`. |
| 27 | Why that decision makes sense | Valuable reasoning semantics but no native visual grounding. |
| 28 | What still must be acquired or verified | Resolve page snapshots/HTML provenance and cap near-duplicate scenarios by group. |

### 6.7 MESAQA

> **Quick take**
> - **Main project role:** `exclude`.
> - **What it uniquely adds:** Methodological anchor for leave-one-span-out verification and evidence cardinality.
> - **Planned original records:** 0 selector/planner records in the current plan.
> - **Why this decision:** Unclear license and synthetic text-only construction are not needed given stronger sources.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **MESAQA: A Dataset for Multi-Span Contextual and Evidence-Grounded Question Answering**. |
| 2 | Where to find the paper and official release | Paper: [MESAQA-P]. Official repository/HF: [MESAQA-GH], [MESAQA-HF]. |
| 3 | Can we get it now? | Public text dataset on HF. |
| 4 | Exact version or revision | COLING 2025 release; repository README SHA recorded in source ledger. |
| 5 | What the license allows — and does not allow | **Still unresolved** (`unresolved`) No clear dataset license was located in the paper/repository/card at cutoff. |
| 6 | Where the original documents came from | Derived from MASH-QA healthcare contexts with LLM question generation and filtering. |
| 7 | What kind of material it contains | Text contexts with sentence evidence IDs; no page images. |
| 8 | How many documents | Context-level count not separately published. |
| 9 | How many pages or images | No images/pages. |
| 10 | How many questions | 6,183 examples (**What we calculated from cited facts** (`derived_calculation`) from the paper's evidence-cardinality table). |
| 11 | Official train, development, and test counts | Exact split manifest was not clearly documented in accessible primary materials. |
| 12 | Kinds of answers | Multi-span extractive answers. |
| 13 | Kinds of reasoning | Aggregation over 2–>5 evidence sentences, consecutive/non-consecutive evidence. |
| 14 | Where and how the evidence is spread | Multiple evidence sentences in one text context. |
| 15 | What evidence labels exist and how precise they are | Evidence strings and `evidence_idx` sentence IDs. |
| 16 | How spatial coordinates are represented | No spatial coordinates. |
| 17 | Are the original PDFs or page images available? | No original page images. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Could be rendered synthetically, but would not test real document layout. |
| 19 | Does it label the complete evidence set? | Intended to be complete at sentence-evidence level; automatic construction remains a quality risk. |
| 20 | Known gaps or quality concerns | LLM generation/filtering, domain-specific healthcare style, unclear license/splits. |
| 21 | Could it overlap with other training or evaluation data? | Potential MASH-QA parent overlap; group by source context. |
| 22 | Real example questions from official material | One exact official paper example is visible; a second exact question was not recoverable from accessible official text. No constructed example is substituted. |
| 23 | What this source uniquely adds | Methodological anchor for leave-one-span-out verification and evidence cardinality. |
| 24 | How many original records this project plans to use | 0 selector/planner records in the current plan. |
| 25 | Planned augmentations and exact quotas | 0. |
| 26 | Its main job in this project — or the decision to exclude it | `exclude`. |
| 27 | Why that decision makes sense | Unclear license and synthetic text-only construction are not needed given stronger sources. |
| 28 | What still must be acquired or verified | Re-audit license and exact split/row manifest before any future use. |

### 6.8 SEMQA / QuoteSum

> **Quick take**
> - **Main project role:** `answerer_sft`.
> - **What it uniquely adds:** Bridge from evidence extraction to attributed synthesis.
> - **Planned original records:** Selector: 0. Answerer-SFT pool: **1,200** released answer rows sampled source-title-disjoint from the 1,428-row train file.
> - **Why this decision:** Human semi-extractive answers directly serve Stage 3 without weakening selector labels.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **SEMQA: Semi-Extractive Multi-Source Question Answering**; released dataset **QuoteSum**. |
| 2 | Where to find the paper and official release | Paper: [QUOTESUM-P]. Official archived repository: [QUOTESUM-GH]. |
| 3 | Can we get it now? | Public JSONL train/dev/test files. |
| 4 | Exact version or revision | QuoteSum v1; repository archived but stable. File SHAs recorded in source ledger. |
| 5 | What the license allows — and does not allow | CC BY-SA 4.0. |
| 6 | Where the original documents came from | Wikipedia source passages paired with human-written semi-extractive answers. |
| 7 | What kind of material it contains | Multi-source text QA; no native document page images. |
| 8 | How many documents | Questions aggregate multiple Wikipedia source pages; document count is not reported as a single corpus total. |
| 9 | How many pages or images | No page images. |
| 10 | How many questions | **What the official source says** (`primary_source_fact`) The paper reports 1,376 questions, 4,009 semi-extractive answers, and 6,451 source paragraphs. **What we directly verified in the release** (`verified_release_observation`) QuoteSum v1 contains exactly **1,428 train + 162 dev + 858 test = 2,448 JSONL answer rows**. [QUOTESUM-P] [QUOTESUM-GH] |
| 11 | Official train, development, and test counts | **What the official source says** (`primary_source_fact`) The paper gives a 60/7/33 question split and 800 test questions. **What we directly verified in the release** (`verified_release_observation`) Release-file answer-row counts are **1,428 train, 162 dev, and 858 test**; these are answer records, not unique questions. [QUOTESUM-P] [QUOTESUM-GH] |
| 12 | Kinds of answers | Semi-extractive synthesis with explicit attributed spans. |
| 13 | Kinds of reasoning | Multi-source aggregation and synthesis. |
| 14 | Where and how the evidence is spread | Multiple documents/source paragraphs. |
| 15 | What evidence labels exist and how precise they are | Inline source-span attribution markers in human answers. |
| 16 | How spatial coordinates are represented | No spatial coordinates. |
| 17 | Are the original PDFs or page images available? | No page images; contexts can be rendered only as synthetic bundles. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Attribution markers can map to spans/candidates in a synthetic rendering. |
| 19 | Does it label the complete evidence set? | Strong at answer-claim attribution, not visual evidence. |
| 20 | Known gaps or quality concerns | Wikipedia context and multi-document setting differ from real same-document QA. |
| 21 | Could it overlap with other training or evaluation data? | Question variants share source sets; split by source-title identity. |
| 22 | Real example questions from official material | Official release rows include “what is the title of tears for fears song?” and “how long did it take to build the temple of jerusalem?” [QUOTESUM-GH] |
| 23 | What this source uniquely adds | Bridge from evidence extraction to attributed synthesis. |
| 24 | How many original records this project plans to use | Selector: 0. Answerer-SFT pool: **1,200** released answer rows sampled source-title-disjoint from the 1,428-row train file. |
| 25 | Planned augmentations and exact quotas | Semi-extractive formatting/attribution only; no selector boxes unless a separately approved synthetic-rendering study is run. |
| 26 | Its main job in this project — or the decision to exclude it | `answerer_sft`. |
| 27 | Why that decision makes sense | Human semi-extractive answers directly serve Stage 3 without weakening selector labels. |
| 28 | What still must be acquired or verified | Reproduce the 1,428/162/858 JSONL line counts, preserve source-title groups, and verify CC BY-SA obligations in derivative answer models. |

### 6.9 M-LongDoc and DocDownstream10K

> **Quick take**
> - **Main project role:** `sealed_evaluation` (primary); `later_rl` is conditional secondary use of a disjoint training corpus.
> - **What it uniquely adds:** Potential later outcome-based RL pool and sealed long-document benchmark.
> - **Planned original records:** Selector: 0. Conditional later-RL target: 10,000 DocDownstream records only after rights approval. Benchmark: 0.
> - **Why this decision:** Long-document realism is useful, but evidence and license are insufficient for selector labels.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **M-LongDoc** benchmark plus **DocDownstream10K-1/-2** training corpora. |
| 2 | Where to find the paper and official release | Paper: [MLONG-P]. Official repository: [MLONG-GH]. |
| 3 | Can we get it now? | Benchmark and metadata publicly listed; no repository LICENSE was found. |
| 4 | Exact version or revision | Repository latest observed commit `944a61950...`; exact full SHA recorded during acquisition. |
| 5 | What the license allows — and does not allow | **Still unresolved** (`unresolved`) No explicit dataset license. Source-document copyrights vary. |
| 6 | Where the original documents came from | Long multimodal documents from public sources; synthetic/curated downstream QA. |
| 7 | What kind of material it contains | Long PDFs with text, tables, figures, and images. |
| 8 | How many documents | Current README: 180 benchmark documents; each DocDownstream variant 300 documents. |
| 9 | How many pages or images | Page count not consistently published in the README; derive from manifest. |
| 10 | How many questions | Current README: 1,051 benchmark questions; 851 high-quality subset; each DocDownstream variant 10,070 questions. |
| 11 | Official train, development, and test counts | Benchmark quality subsets rather than standard train/dev/test; DocDownstream-1/-2 are separate training corpora. |
| 12 | Kinds of answers | Open-ended/extractive/abstractive long-document answers. |
| 13 | Kinds of reasoning | Long-context retrieval, figure/table reasoning, explanation. |
| 14 | Where and how the evidence is spread | Potentially multi-page, but evidence labels are not uniformly complete. |
| 15 | What evidence labels exist and how precise they are | Question/answer and document context; benchmark evidence supervision is weaker/noisier than selector gold. |
| 16 | How spatial coordinates are represented | No uniform native boxes. |
| 17 | Are the original PDFs or page images available? | PDFs may be linked/packaged subject to source rights. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Possible only through pseudo-evidence generation. |
| 19 | Does it label the complete evidence set? | No reliable general complete-evidence guarantee. |
| 20 | Known gaps or quality concerns | Paper/current README count drift (earlier 851 versus current 1,051); noisy contexts and licensing uncertainty. |
| 21 | Could it overlap with other training or evaluation data? | Potential overlap with public benchmark documents and model pretraining. |
| 22 | Real example questions from official material | Paper examples include “What is Omnivore proven to be empirically similar to?”; a second exact example was not reliably recoverable from accessible official text. |
| 23 | What this source uniquely adds | Potential later outcome-based RL pool and sealed long-document benchmark. |
| 24 | How many original records this project plans to use | Selector: 0. Conditional later-RL target: 10,000 DocDownstream records only after rights approval. Benchmark: 0. |
| 25 | Planned augmentations and exact quotas | Outcome-only RL; never selector SFT without verified pseudo-evidence. |
| 26 | Its main job in this project — or the decision to exclude it | `sealed_evaluation` (primary); `later_rl` is conditional secondary use of a disjoint training corpus. |
| 27 | Why that decision makes sense | Long-document realism is useful, but evidence and license are insufficient for selector labels. |
| 28 | What still must be acquired or verified | Obtain license, freeze README/revision, reconcile 851/1,051, and verify source-document redistribution. |

### 6.10 MultiModalQA

> **Quick take**
> - **Main project role:** `planner_or_decomposition_pretraining`.
> - **What it uniquely adds:** Multi-modal composition and explicit synthetic-bundle methodology.
> - **Planned original records:** Selector: 0. Planner pool: 2,000; answerer pool: 4,000 disjoint; conditional RL: 4,000 disjoint.
> - **Why this decision:** Useful for composition but not fair as native real-document selector evidence.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **MultiModalQA: Complex Question Answering over Text, Tables and Images**. |
| 2 | Where to find the paper and official release | Paper: [MMQA-P]. Official repository: [MMQA-GH]. |
| 3 | Can we get it now? | Public annotations and linked multimodal contexts. |
| 4 | Exact version or revision | Official 2021 release. |
| 5 | What the license allows — and does not allow | Use official repository/data terms; source Wikipedia/images retain their licenses. |
| 6 | Where the original documents came from | Wikipedia text, tables, and images linked by entities. |
| 7 | What kind of material it contains | Multi-document multimodal QA, not a single PDF. |
| 8 | How many documents | Context pool is entity/source based; use release manifest for unique-source counts. |
| 9 | How many pages or images | Images/tables/text contexts rather than pages. |
| 10 | How many questions | 29,918 questions. |
| 11 | Official train, development, and test counts | 23,817 train; 2,441 dev; 3,660 test. |
| 12 | Kinds of answers | Extractive, numerical, yes/no, and composed answers. |
| 13 | Kinds of reasoning | Bridge/comparison, image-text/table-text composition, entity resolution. |
| 14 | Where and how the evidence is spread | Multiple documents/modalities; explicit multi-source evidence. |
| 15 | What evidence labels exist and how precise they are | Supporting context IDs and decomposition/composition structure, but no page boxes. |
| 16 | How spatial coordinates are represented | No document-page coordinate system. |
| 17 | Are the original PDFs or page images available? | Linked source assets available subject to source terms. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Can map to synthetic bundle candidates, not real same-document pages. |
| 19 | Does it label the complete evidence set? | Strong at source-level evidence, weaker at fine spatial completeness. |
| 20 | Known gaps or quality concerns | Entity-linked construction and synthetic composition may be less natural than reader-authored document questions. |
| 21 | Could it overlap with other training or evaluation data? | M3DocVQA renders/supports MultiModalQA questions; prevent train–evaluation leakage. |
| 22 | Real example questions from official material | Official paper examples include “When was the player who was drafted by the Memphis Grizzlies born?” and “What was the population of the country whose flag is shown?” [MMQA-P] |
| 23 | What this source uniquely adds | Multi-modal composition and explicit synthetic-bundle methodology. |
| 24 | How many original records this project plans to use | Selector: 0. Planner pool: 2,000; answerer pool: 4,000 disjoint; conditional RL: 4,000 disjoint. |
| 25 | Planned augmentations and exact quotas | Question decomposition/composition and answerer training only; synthetic-bundle selector records are generated under §11 and parent IDs are preserved. |
| 26 | Its main job in this project — or the decision to exclude it | `planner_or_decomposition_pretraining`. |
| 27 | Why that decision makes sense | Useful for composition but not fair as native real-document selector evidence. |
| 28 | What still must be acquired or verified | Create disjoint objective pools and quarantine every question used by M3DocVQA evaluation. |

### 6.11 HotpotQA

> **Quick take**
> - **Main project role:** `planner_or_decomposition_pretraining`.
> - **What it uniquely adds:** Large bridge/comparison decomposition source.
> - **Planned original records:** Selector: 0. Planner pool: 2,000 source-title-disjoint train questions.
> - **Why this decision:** Useful reasoning structure without pretending text paragraphs are document regions.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering**. |
| 2 | Where to find the paper and official release | Paper: [HOTPOT-P]. Official site/release: [HOTPOT-SITE]. |
| 3 | Can we get it now? | Public text QA dataset. |
| 4 | Exact version or revision | Original fullwiki/distractor releases. |
| 5 | What the license allows — and does not allow | CC BY-SA 4.0 annotations; Wikipedia source content licenses apply. |
| 6 | Where the original documents came from | Wikipedia paragraphs assembled into two-hop questions with distractors. |
| 7 | What kind of material it contains | Text-only multi-document QA. |
| 8 | How many documents | Article/context counts vary by release configuration. |
| 9 | How many pages or images | No images/pages. |
| 10 | How many questions | Approximately 113k questions. |
| 11 | Official train, development, and test counts | 90,447 train; development/test totals differ by release configuration, so acquisition manifest is authoritative. Only train is used. |
| 12 | Kinds of answers | Extractive and yes/no. |
| 13 | Kinds of reasoning | Bridge and comparison; supporting-fact retrieval. |
| 14 | Where and how the evidence is spread | Multiple documents and supporting sentences. |
| 15 | What evidence labels exist and how precise they are | Native supporting sentence labels. |
| 16 | How spatial coordinates are represented | No spatial coordinates. |
| 17 | Are the original PDFs or page images available? | No document images. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Can support planner/decomposition only or synthetic rendered bundles. |
| 19 | Does it label the complete evidence set? | Good sentence evidence but some supporting-fact labels are not minimal/complete. |
| 20 | Known gaps or quality concerns | Wikipedia style and template leakage; not real same-document visual QA. |
| 21 | Could it overlap with other training or evaluation data? | Potential overlap with MultiModalQA/QuoteSum/Wikipedia-based evaluations. |
| 22 | Real example questions from official material | Official examples: “Which magazine was started first, Arthur's Magazine or First for Women?” and “What government position was held by the woman who portrayed Corliss Archer?” [HOTPOT-P] |
| 23 | What this source uniquely adds | Large bridge/comparison decomposition source. |
| 24 | How many original records this project plans to use | Selector: 0. Planner pool: 2,000 source-title-disjoint train questions. |
| 25 | Planned augmentations and exact quotas | Planner subgoals only; no selector boxes. |
| 26 | Its main job in this project — or the decision to exclude it | `planner_or_decomposition_pretraining`. |
| 27 | Why that decision makes sense | Useful reasoning structure without pretending text paragraphs are document regions. |
| 28 | What still must be acquired or verified | Pin the exact release/config and remove source-title overlap with every Wikipedia-derived evaluation. |

### 6.12 BREAK / QDMR

> **Quick take**
> - **Main project role:** `planner_or_decomposition_pretraining`.
> - **What it uniquely adds:** Explicit decomposition syntax for planner pretraining.
> - **Planned original records:** Selector: 0. Planner pool: 1,500 train decompositions.
> - **Why this decision:** Provides decomposition supervision only; avoids contaminating spatial labels.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **BREAK: a Question Decomposition Meaning Representation dataset**. |
| 2 | Where to find the paper and official release | Paper: [BREAK-P]. Official repository: [BREAK-GH]. |
| 3 | Can we get it now? | Public decomposition annotations. |
| 4 | Exact version or revision | Official release used by the paper; exact manifest pinned at acquisition. |
| 5 | What the license allows — and does not allow | Use repository/data license stated in the acquired release; do not infer data rights from code. |
| 6 | Where the original documents came from | Questions aggregated from multiple QA datasets with QDMR decompositions. |
| 7 | What kind of material it contains | Text questions/decompositions; no documents or evidence images. |
| 8 | How many documents | Not applicable as a document corpus. |
| 9 | How many pages or images | No pages/images. |
| 10 | How many questions | 83,978 questions. |
| 11 | Official train, development, and test counts | Common release counts: 44,321 train; 7,760 dev; 31,897 test; verify manifest before use. |
| 12 | Kinds of answers | Inherited answer types; the release focuses on decompositions. |
| 13 | Kinds of reasoning | Compositional operators, filtering, projection, comparison, aggregation, arithmetic. |
| 14 | Where and how the evidence is spread | No native evidence topology. |
| 15 | What evidence labels exist and how precise they are | QDMR step sequences, not supporting evidence. |
| 16 | How spatial coordinates are represented | No coordinates. |
| 17 | Are the original PDFs or page images available? | No document assets. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Not applicable to candidates; used only to initialize planner/subgoal representations. |
| 19 | Does it label the complete evidence set? | No evidence annotation. |
| 20 | Known gaps or quality concerns | Decompositions can be ambiguous and may not reflect visual-document operations. |
| 21 | Could it overlap with other training or evaluation data? | Parent-dataset overlap; group by original dataset/question ID. |
| 22 | Real example questions from official material | Official examples include “Who is the father of the wife of the current president of the US?” and “What are the names of works by Ayn Rand that were adapted to film?” [BREAK-P] |
| 23 | What this source uniquely adds | Explicit decomposition syntax for planner pretraining. |
| 24 | How many original records this project plans to use | Selector: 0. Planner pool: 1,500 train decompositions. |
| 25 | Planned augmentations and exact quotas | No selector augmentation; operators may condition the generator/verifier prompts. |
| 26 | Its main job in this project — or the decision to exclude it | `planner_or_decomposition_pretraining`. |
| 27 | Why that decision makes sense | Provides decomposition supervision only; avoids contaminating spatial labels. |
| 28 | What still must be acquired or verified | Pin license/release and normalize operator vocabulary without exposing QDMR to selector inference unless ablated. |


## 7. Extra datasets discovered along the way

**Friendly guidepost.** The seed list was not the finish line. This section records additional datasets found during the review and asks one practical question for each: does it close a real gap better than the sources already selected?

Additional sources were included only when they materially filled a gap, exposed a contamination risk, or changed the legal/evaluation plan. TAT-DQA, PeerQA, and RefChartQA enter the preferred corpus; the remaining sources are auxiliary, method anchors, or sealed evaluation.

### 7.1 TAT-DQA

> **Quick take**
> - **Main project role:** `core_selector_sft`.
> - **What it uniquely adds:** Best currently licensed visual numerical/table-text anchor.
> - **Planned original records:** Minimum 5,000; full 12,000 train questions.
> - **Why this decision:** Combines real page images with evidence/program supervision.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | Canonical details are in §6.4/§7.1 additional discovery; TAT-DQA is the visual extension of TAT-QA. |
| 2 | Where to find the paper and official release | [TATDQA-P], [TATDQA-GH], [TATDQA-HF]. |
| 3 | Can we get it now? | Public release and images/OCR available. |
| 4 | Exact version or revision | Paper v3; official repository master snapshot. |
| 5 | What the license allows — and does not allow | CC BY 4.0 dataset release; source-report copyrights apply. |
| 6 | Where the original documents came from | AnnualReports financial pages corresponding to TAT-QA contexts. |
| 7 | What kind of material it contains | One-to-three page financial documents with tables/text/OCR/layout. |
| 8 | How many documents | 2,758 documents from 182 reports. |
| 9 | How many pages or images | 3,067 pages. |
| 10 | How many questions | 16,558 questions. |
| 11 | Official train, development, and test counts | 13,251 train; 1,645 dev; 1,662 test. |
| 12 | Kinds of answers | Span 7,141; multi-span 2,082; count 377; arithmetic 6,958. |
| 13 | Kinds of reasoning | Arithmetic, counting, comparison, table-text, header dependencies. |
| 14 | Where and how the evidence is spread | One or multiple regions; 11%+ documents multi-page; some real cross-page potential. |
| 15 | What evidence labels exist and how precise they are | Word/block boxes, evidence tags, answer derivations/programs. |
| 16 | How spatial coordinates are represented | Native PDF/OCR pixel boxes; normalize to page dimensions. |
| 17 | Are the original PDFs or page images available? | Yes. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | High, provided table rows/headers are not fragmented beyond oracle repair. |
| 19 | Does it label the complete evidence set? | Programs/evidence are strong; structural dependencies still require completeness verification. |
| 20 | Known gaps or quality concerns | Paper notes the baseline often selects the major table page for multi-page documents, so “multi-page document” does not guarantee cross-page evidence. |
| 21 | Could it overlap with other training or evaluation data? | Overlaps TAT-QA questions; source reports can overlap FinQA/MultiHiertt/DocFinQA. |
| 22 | Real example questions from official material | Official example: “What was the total cost in Wireless including spectrum license fee in 2019?” A second exact official release question was not accessible without acquiring the annotation file; no constructed substitute is used. [TATDQA-P] |
| 23 | What this source uniquely adds | Best currently licensed visual numerical/table-text anchor. |
| 24 | How many original records this project plans to use | Minimum 5,000; full 12,000 train questions. |
| 25 | Planned augmentations and exact quotas | Minimum 3,700 accepted; full family targets include 2,500 same-page, 3,500 real cross-page, 1,600 composition, 1,800 semi-extractive, 1,500 abstractive, and adversarial variants. |
| 26 | Its main job in this project — or the decision to exclude it | `core_selector_sft`. |
| 27 | Why that decision makes sense | Combines real page images with evidence/program supervision. |
| 28 | What still must be acquired or verified | Reproduce page/report hashes, quantify genuine cross-page subset, and verify every operand/header maps to frozen candidates. |

### 7.2 PeerQA

> **Quick take**
> - **Main project role:** `auxiliary_selector_sft`.
> - **What it uniquely adds:** Real expert information needs and explanatory/limitation questions.
> - **Planned original records:** Minimum 0; full 400 aligned records.
> - **Why this decision:** Provides realism absent from synthetic scientific QA, with manageable manual audit scale.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **PeerQA: A Scientific Question Answering Dataset from Peer Reviews**. |
| 2 | Where to find the paper and official release | Paper: [PEERQA-P]. Official repository: [PEERQA-GH]. |
| 3 | Can we get it now? | Public v1.0 release. |
| 4 | Exact version or revision | Latest observed main commit dated 2025-06-23. |
| 5 | What the license allows — and does not allow | CC BY-NC-SA 4.0. |
| 6 | Where the original documents came from | Rejected or questioned research-paper claims drawn from peer reviews; corresponding papers and author answers. |
| 7 | What kind of material it contains | Long scientific papers represented as text with evidence; `visual_info` metadata exists. |
| 8 | How many documents | 208 papers. |
| 9 | How many pages or images | Page count derived after PDF acquisition. |
| 10 | How many questions | 579 questions; 6,823 relevance annotations; 2,020 answers; 593 contributing authors. |
| 11 | Official train, development, and test counts | No official train/dev/test. Paper evaluation includes 216 answerable and 72 unanswerable test questions; this project creates a paper-disjoint 90/5/5 grouping before augmentation. |
| 12 | Kinds of answers | Extractive, abstractive, explanatory, and unanswerable. |
| 13 | Kinds of reasoning | Critique, method/result interpretation, counterfactual/limitation, explanation. |
| 14 | Where and how the evidence is spread | Multiple evidence passages within a paper; potentially cross-page. |
| 15 | What evidence labels exist and how precise they are | Passage relevance annotations and answers. |
| 16 | How spatial coordinates are represented | No native boxes; derive after PDF alignment. |
| 17 | Are the original PDFs or page images available? | Paper text/data available; PDF rights checked per paper. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | High after paragraph/page alignment; visuals referenced through `visual_info` need manual verification. |
| 19 | Does it label the complete evidence set? | Passage relevance may be broad rather than minimal. |
| 20 | Known gaps or quality concerns | Small size, repeated review language, paper availability/licensing variation. |
| 21 | Could it overlap with other training or evaluation data? | Potential paper overlap with QASPER/SciEGQA/SPIQA. |
| 22 | Real example questions from official material | Official examples include “Could supervised agents outperform the reported performance?” and “Why is memorization not effective?” [PEERQA-P] |
| 23 | What this source uniquely adds | Real expert information needs and explanatory/limitation questions. |
| 24 | How many original records this project plans to use | Minimum 0; full 400 aligned records. |
| 25 | Planned augmentations and exact quotas | Full: 100 paraphrases, 100 compositions, 500 real-cross-page candidates accepted across original/derived pool, 200 synthetic bundles, 600 semi-extractive, 500 abstractive, 100 missing, 100 query negatives; record counts are target augmentations, not unique parents. |
| 26 | Its main job in this project — or the decision to exclude it | `auxiliary_selector_sft`. |
| 27 | Why that decision makes sense | Provides realism absent from synthetic scientific QA, with manageable manual audit scale. |
| 28 | What still must be acquired or verified | Create official-release snapshot, legal PDF-acquisition ledger, and paper-disjoint custom split before any generation. |

### 7.3 RefChartQA

> **Quick take**
> - **Main project role:** `auxiliary_selector_sft`.
> - **What it uniquely adds:** Best available chart-element grounding source.
> - **Planned original records:** Minimum 1,000; full 6,000 train records, conditional on legal approval.
> - **Why this decision:** Fills chart/legend/reference gap not covered by document OCR boxes.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **RefChartQA: Grounding Visual References in Chart Question Answering**. |
| 2 | Where to find the paper and official release | Paper: [REFCHART-P]. Official repository/HF: [REFCHART-GH], [REFCHART-HF]. |
| 3 | Can we get it now? | Public train/validation/test data. |
| 4 | Exact version or revision | Current official release; repository commit recorded in source ledger. |
| 5 | What the license allows — and does not allow | License conflict: repository GPL-3.0 and HF AGPL-3.0. Apply AGPL-3.0 plus legal review; parent ChartQA rights still apply. |
| 6 | Where the original documents came from | Chart images and questions derived from ChartQA with element references and grounding. |
| 7 | What kind of material it contains | Bar/line/pie charts with text/element boxes. |
| 8 | How many documents | Chart count requires acquired manifest because questions outnumber charts. |
| 9 | How many pages or images | Chart images packaged/referenced. |
| 10 | How many questions | 73,702 rows: 55,789 train; 6,223 validation; 11,690 test. |
| 11 | Official train, development, and test counts | As above. |
| 12 | Kinds of answers | Extractive, categorical, numerical, and reference-following answers. |
| 13 | Kinds of reasoning | Reference resolution, comparison, ordering, arithmetic, iterative/multi-hop. |
| 14 | Where and how the evidence is spread | Multiple chart elements in one image. |
| 15 | What evidence labels exist and how precise they are | Grounded chart-element boxes and task labels: answer-referring 57,179; iterative 11,107; multi-hop 5,379; textual-query 37. |
| 16 | How spatial coordinates are represented | Image pixel boxes. |
| 17 | Are the original PDFs or page images available? | Chart images available. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | High for chart elements; parser needs atomic bars/labels/legend candidates. |
| 19 | Does it label the complete evidence set? | Not automatically: axes, scale, legend, or denominator elements can be necessary but unlabeled. |
| 20 | Known gaps or quality concerns | License conflict and derivative relationship to ChartQA; some reference language may be template-like. |
| 21 | Could it overlap with other training or evaluation data? | Direct overlap with ChartQA images/questions. |
| 22 | Real example questions from official material | Official examples: “Referring to plotly purple, return the category with the lowest value.” and “What is the percentage of the third bar from the right?” [REFCHART-P] [REFCHART-GH] |
| 23 | What this source uniquely adds | Best available chart-element grounding source. |
| 24 | How many original records this project plans to use | Minimum 1,000; full 6,000 train records, conditional on legal approval. |
| 25 | Planned augmentations and exact quotas | Minimum 500 accepted; full includes 1,300 same-page, 600 composition, 700 synthetic bundles, 300 semi-extractive, 300 abstractive, 500 missing, 500 query negatives. |
| 26 | Its main job in this project — or the decision to exclude it | `auxiliary_selector_sft`. |
| 27 | Why that decision makes sense | Fills chart/legend/reference gap not covered by document OCR boxes. |
| 28 | What still must be acquired or verified | Obtain legal approval, pin exact release counts, detect ChartQA duplicates, and measure complete header/legend oracle recall. |

### 7.4 SPIQA

> **Quick take**
> - **Main project role:** `answerer_sft`.
> - **What it uniquely adds:** Large answerer-SFT and later-RL source for chart/figure explanations.
> - **Planned original records:** Selector: 0. Answerer pool: **6,800**; conditional RL: 6,000 disjoint.
> - **Why this decision:** Adds scientific visual reasoning without contaminating selector supervision.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **SPIQA: A Dataset for Multimodal Question Answering on Scientific Papers**. |
| 2 | Where to find the paper and official release | Paper: [SPIQA-P]. Official HF release: [SPIQA-HF]. |
| 3 | Can we get it now? | Public dataset. |
| 4 | Exact version or revision | NeurIPS 2024 release; HF revision pinned at acquisition. |
| 5 | What the license allows — and does not allow | Use official HF terms/source-paper licenses; no broad redistribution assumption. |
| 6 | Where the original documents came from | Scientific papers with figures, tables, captions, and surrounding text. |
| 7 | What kind of material it contains | Multimodal scientific QA centered on reference figures/tables. |
| 8 | How many documents | Paper pool 25,859; test covers 1,201 papers. |
| 9 | How many pages or images | Figure/table assets rather than complete page-image manifests. |
| 10 | How many questions | Approximately 270k generated train questions; 11,820 manually verified test questions. |
| 11 | Official train, development, and test counts | Large generated train; manually verified test-A/test-B partitions. |
| 12 | Kinds of answers | Extractive and explanatory scientific answers. |
| 13 | Kinds of reasoning | Figure/table interpretation, caption/context synthesis, comparison. |
| 14 | Where and how the evidence is spread | One or more reference figures/tables plus text; not necessarily page-grounded. |
| 15 | What evidence labels exist and how precise they are | Reference image IDs and text/caption context; no complete page boxes. |
| 16 | How spatial coordinates are represented | Figure/image coordinates, not uniform page coordinates. |
| 17 | Are the original PDFs or page images available? | Paper/figure assets available under source terms. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Can map reference images/captions to candidates after paper reacquisition, but training evidence remains generated. |
| 19 | Does it label the complete evidence set? | No complete page-level guarantee. |
| 20 | Known gaps or quality concerns | Generated training questions may have teacher/style artifacts; paper overlap contamination is substantial. |
| 21 | Could it overlap with other training or evaluation data? | Potential overlap with QASPER, PeerQA, SciEGQA, DocScope, and arXiv benchmarks. |
| 22 | Real example questions from official material | Official paper provides illustrated scientific questions; two exact strings were not reliably available in the accessible text layer, so no constructed examples are substituted. |
| 23 | What this source uniquely adds | Large answerer-SFT and later-RL source for chart/figure explanations. |
| 24 | How many original records this project plans to use | Selector: 0. Answerer pool: **6,800**; conditional RL: 6,000 disjoint. |
| 25 | Planned augmentations and exact quotas | No selector labels. Use only answer generation with source references; evidence pseudo-labeling requires a future addendum. |
| 26 | Its main job in this project — or the decision to exclude it | `answerer_sft`. |
| 27 | Why that decision makes sense | Adds scientific visual reasoning without contaminating selector supervision. |
| 28 | What still must be acquired or verified | Pin paper IDs, remove overlap with scientific evaluations, and verify source-paper licenses. |

### 7.5 HiTab

> **Quick take**
> - **Main project role:** `planner_or_decomposition_pretraining`.
> - **What it uniquely adds:** Hierarchical-header planning and operand selection.
> - **Planned original records:** Selector: 0. Planner pool: 5,000 train records.
> - **Why this decision:** Unique hierarchical-table supervision without claiming native page grounding.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **HiTab: A Hierarchical Table Dataset for Question Answering and Natural Language Generation**. |
| 2 | Where to find the paper and official release | Paper: [HITAB-P]. Official repository: [HITAB-GH]. |
| 3 | Can we get it now? | Public data/code. |
| 4 | Exact version or revision | Current official release. |
| 5 | What the license allows — and does not allow | MIT. |
| 6 | Where the original documents came from | Hierarchical tables from Wikipedia/statistical sources. |
| 7 | What kind of material it contains | Structured tables and text questions; no native page images. |
| 8 | How many documents | 3,597 tables. |
| 9 | How many pages or images | No page images. |
| 10 | How many questions | Release README 10,672 QA; paper abstract 10,686. Use release manifest. |
| 11 | Official train, development, and test counts | Table-disjoint 70/15/15 split; test has 1,584 questions; exact train/dev counts from acquired manifest. |
| 12 | Kinds of answers | Extractive/numerical/program-derived. |
| 13 | Kinds of reasoning | Hierarchical header traversal, aggregation, comparison, arithmetic. |
| 14 | Where and how the evidence is spread | Multiple cells/header paths within one table. |
| 15 | What evidence labels exist and how precise they are | Logical forms/programs and cell/header structure. |
| 16 | How spatial coordinates are represented | Table cell coordinates, not pixels. |
| 17 | Are the original PDFs or page images available? | No native document pages. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Can generate synthetic rendered tables, but visual layout would be artificial. |
| 19 | Does it label the complete evidence set? | Strong structural evidence; visual completeness not native. |
| 20 | Known gaps or quality concerns | Published count conflict and synthetic table rendering risk. |
| 21 | Could it overlap with other training or evaluation data? | Potential overlap with Wikipedia tables used by MultiModalQA/Chart sources. |
| 22 | Real example questions from official material | Official paper figures contain authentic questions; two exact strings should be extracted during acquisition because the accessible text layer did not preserve them reliably. |
| 23 | What this source uniquely adds | Hierarchical-header planning and operand selection. |
| 24 | How many original records this project plans to use | Selector: 0. Planner pool: 5,000 train records. |
| 25 | Planned augmentations and exact quotas | No direct selector augmentation; programs may seed verified header-dependency constructions. |
| 26 | Its main job in this project — or the decision to exclude it | `planner_or_decomposition_pretraining`. |
| 27 | Why that decision makes sense | Unique hierarchical-table supervision without claiming native page grounding. |
| 28 | What still must be acquired or verified | Reconcile 10,672/10,686 and extract exact split counts/question examples from the pinned manifest. |

### 7.6 ChartQA

> **Quick take**
> - **Main project role:** `exclude`.
> - **What it uniquely adds:** Parent provenance for RefChartQA.
> - **Planned original records:** 0 minimum; 0 full.
> - **Why this decision:** RefChartQA supplies stronger boxes on overlapping chart assets.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning**. |
| 2 | Where to find the paper and official release | Paper: [CHARTQA-P]. Official repository: [CHARTQA-GH]. |
| 3 | Can we get it now? | Public data/code. |
| 4 | Exact version or revision | Official release. |
| 5 | What the license allows — and does not allow | GPL-3.0 code; dataset/image rights are less clear and must be reviewed separately. |
| 6 | Where the original documents came from | Real and generated charts with human and machine questions. |
| 7 | What kind of material it contains | Chart images plus data tables. |
| 8 | How many documents | 20,882 charts. |
| 9 | How many pages or images | 20,882 images. |
| 10 | How many questions | 32,719 questions: 9,608 human and 23,111 machine-generated. |
| 11 | Official train, development, and test counts | Train 18,317 charts/28,299 Q; val 1,056/1,920; test 1,509/2,500. |
| 12 | Kinds of answers | Numerical, categorical, yes/no. |
| 13 | Kinds of reasoning | Arithmetic, comparison, visual lookup, trends. |
| 14 | Where and how the evidence is spread | Multiple chart elements in one image. |
| 15 | What evidence labels exist and how precise they are | Question/answer and underlying chart data; no native evidence boxes. |
| 16 | How spatial coordinates are represented | No evidence coordinates. |
| 17 | Are the original PDFs or page images available? | Chart images/data available. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Could pseudo-align, but RefChartQA is the cleaner grounded derivative. |
| 19 | Does it label the complete evidence set? | No. |
| 20 | Known gaps or quality concerns | Machine-generated templates and incomplete spatial supervision. |
| 21 | Could it overlap with other training or evaluation data? | RefChartQA directly derives from ChartQA; using both duplicates images/questions. |
| 22 | Real example questions from official material | Paper/template examples include “What is the difference between the highest and the lowest value?” and binary comparison questions; exact wording varies by generated template and should be taken from the acquired release. |
| 23 | What this source uniquely adds | Parent provenance for RefChartQA. |
| 24 | How many original records this project plans to use | 0 minimum; 0 full. |
| 25 | Planned augmentations and exact quotas | 0. |
| 26 | Its main job in this project — or the decision to exclude it | `exclude`. |
| 27 | Why that decision makes sense | RefChartQA supplies stronger boxes on overlapping chart assets. |
| 28 | What still must be acquired or verified | Keep parent image/question hashes and resolve image licenses for RefChartQA use. |

### 7.7 LongDocURL

> **Quick take**
> - **Main project role:** `sealed_evaluation`.
> - **What it uniquely adds:** External test of long-document and document-wide tasks.
> - **Planned original records:** 0.
> - **Why this decision:** Primarily a benchmark and lacks selector-grade evidence.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **LongDocURL: Long Document Understanding and Reasoning with Large Language Models**. |
| 2 | Where to find the paper and official release | Paper: [LONGDOCURL-P]. Official project: [LONGDOCURL-PROJ]. |
| 3 | Can we get it now? | Public benchmark/project assets subject to terms. |
| 4 | Exact version or revision | ACL 2025 release. |
| 5 | What the license allows — and does not allow | Use official benchmark and source-document terms; no training permission assumed. |
| 6 | Where the original documents came from | 396 long PDFs from diverse web sources. |
| 7 | What kind of material it contains | Long multimodal PDFs. |
| 8 | How many documents | 396 PDFs. |
| 9 | How many pages or images | More than 33,000 pages; average approximately 85.6 pages/document. |
| 10 | How many questions | 2,325 questions across 20 subtasks. |
| 11 | Official train, development, and test counts | Benchmark partitions/subtasks; no training allocation. |
| 12 | Kinds of answers | Extractive, numerical, explanatory, document-wide. |
| 13 | Kinds of reasoning | Long-context lookup, aggregation, global reasoning. |
| 14 | Where and how the evidence is spread | Single/multiple pages and document-wide scope. |
| 15 | What evidence labels exist and how precise they are | Question/answer and task metadata; evidence granularity varies and is not a complete selector label. |
| 16 | How spatial coordinates are represented | No uniform native boxes. |
| 17 | Are the original PDFs or page images available? | PDFs/links available subject to source terms. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Useful only for sealed evaluation after schema adapter. |
| 19 | Does it label the complete evidence set? | No guaranteed complete region evidence. |
| 20 | Known gaps or quality concerns | Benchmark size and public exposure create contamination risk. |
| 21 | Could it overlap with other training or evaluation data? | Potential overlap with M-LongDoc/MMDocRAG/DocScope public PDFs. |
| 22 | Real example questions from official material | Two exact official example strings were not recoverable from the accessible paper/project text; no constructed examples are substituted. |
| 23 | What this source uniquely adds | External test of long-document and document-wide tasks. |
| 24 | How many original records this project plans to use | 0. |
| 25 | Planned augmentations and exact quotas | 0; no augmentation seeds. |
| 26 | Its main job in this project — or the decision to exclude it | `sealed_evaluation`. |
| 27 | Why that decision makes sense | Primarily a benchmark and lacks selector-grade evidence. |
| 28 | What still must be acquired or verified | Hash every PDF against all training sources and keep questions inaccessible until final evaluation. |

### 7.8 M3DocVQA / M3DocRAG benchmark

> **Quick take**
> - **Main project role:** `sealed_evaluation`.
> - **What it uniquely adds:** External test of multi-document visual retrieval/QA.
> - **Planned original records:** 0.
> - **Why this decision:** Direct MultiModalQA overlap and benchmark purpose preclude training.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **M3DocVQA**, released with **M3DocRAG**. |
| 2 | Where to find the paper and official release | Paper/project: [M3DOC-P], [M3DOC-PROJ]. |
| 3 | Can we get it now? | Public benchmark/project. |
| 4 | Exact version or revision | Official release snapshot. |
| 5 | What the license allows — and does not allow | Use repository/data terms and source Wikipedia licenses. |
| 6 | Where the original documents came from | Rendered Wikipedia pages supporting MultiModalQA-derived questions. |
| 7 | What kind of material it contains | Multi-document, multi-page rendered visual QA. |
| 8 | How many documents | 3,368 PDFs. |
| 9 | How many pages or images | More than 41,005 pages. |
| 10 | How many questions | 2,441 multi-hop questions. |
| 11 | Official train, development, and test counts | Evaluation benchmark. |
| 12 | Kinds of answers | Inherited extractive, numerical, categorical, and composed answers. |
| 13 | Kinds of reasoning | Multi-modal bridge/comparison and retrieval. |
| 14 | Where and how the evidence is spread | Multiple documents/pages. |
| 15 | What evidence labels exist and how precise they are | Supporting-document/page structure inherited from MultiModalQA; no complete fine-region gold. |
| 16 | How spatial coordinates are represented | Rendered-page coordinates if generated; not native region evidence. |
| 17 | Are the original PDFs or page images available? | Rendered PDFs available. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Compatible for evaluation adapter. |
| 19 | Does it label the complete evidence set? | No fine-grained completeness guarantee. |
| 20 | Known gaps or quality concerns | Questions are inherited/composed; rendering is synthetic. |
| 21 | Could it overlap with other training or evaluation data? | Direct overlap with MultiModalQA questions. Any overlapping train question must be removed from planner/answerer pools. |
| 22 | Real example questions from official material | Use authentic MultiModalQA examples in §6.10; M3DocVQA inherits that source family. |
| 23 | What this source uniquely adds | External test of multi-document visual retrieval/QA. |
| 24 | How many original records this project plans to use | 0. |
| 25 | Planned augmentations and exact quotas | 0. |
| 26 | Its main job in this project — or the decision to exclude it | `sealed_evaluation`. |
| 27 | Why that decision makes sense | Direct MultiModalQA overlap and benchmark purpose preclude training. |
| 28 | What still must be acquired or verified | Obtain exact question-ID overlap mapping before selecting any MultiModalQA training pool. |

### 7.9 MMDocIR

> **Quick take**
> - **Main project role:** `sealed_evaluation`.
> - **What it uniquely adds:** External retrieval benchmark and overlap-warning source.
> - **Planned original records:** 0.
> - **Why this decision:** Training mixture would destroy source control and evaluation quarantine.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **MMDocIR: Benchmarking Multi-Modal Retrieval for Long Documents**. |
| 2 | Where to find the paper and official release | Paper: [MMDOCIR-P]. Official repository: [MMDOCIR-GH]. |
| 3 | Can we get it now? | Public benchmark/training mixture. |
| 4 | Exact version or revision | Official 2025 release. |
| 5 | What the license allows — and does not allow | Per-source licenses vary; mixture license cannot supersede parents. |
| 6 | Where the original documents came from | Combines MP-DocVQA, SlideVQA, TAT-DQA, ArXivQA, SciQAG, DUDE, CUAD, and benchmark sources. |
| 7 | What kind of material it contains | Long-document page/image retrieval. |
| 8 | How many documents | 1,685 documents. |
| 9 | How many pages or images | 173,843 pages/images. |
| 10 | How many questions | 73,843 train QA over 987 docs/100k images; 313 expert test QA over 698 docs. |
| 11 | Official train, development, and test counts | Train and expert test. |
| 12 | Kinds of answers | Parent-dependent. |
| 13 | Kinds of reasoning | Page retrieval across long documents. |
| 14 | Where and how the evidence is spread | Page labels and source-specific evidence. |
| 15 | What evidence labels exist and how precise they are | Page-level retrieval supervision. |
| 16 | How spatial coordinates are represented | Page IDs, not region boxes. |
| 17 | Are the original PDFs or page images available? | Assets linked/packaged subject to parents. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Useful as a benchmark adapter, not clean selector training. |
| 19 | Does it label the complete evidence set? | No complete region evidence. |
| 20 | Known gaps or quality concerns | Opaque bootstrapping, heterogeneous schemas, and parent licenses. |
| 21 | Could it overlap with other training or evaluation data? | Massive overlap with DUDE, TAT-DQA, MP-DocVQA, SlideVQA; expert evaluation draws from public benchmarks including MMLongBench-Doc/DocBench. |
| 22 | Real example questions from official material | Exact examples should be read from the sealed expert set only at final evaluation; they are deliberately not reproduced here. |
| 23 | What this source uniquely adds | External retrieval benchmark and overlap-warning source. |
| 24 | How many original records this project plans to use | 0. |
| 25 | Planned augmentations and exact quotas | 0. |
| 26 | Its main job in this project — or the decision to exclude it | `sealed_evaluation`. |
| 27 | Why that decision makes sense | Training mixture would destroy source control and evaluation quarantine. |
| 28 | What still must be acquired or verified | Acquire only benchmark manifests for hash comparison; keep expert questions sealed. |

### 7.10 MMDocRAG

> **Quick take**
> - **Main project role:** `sealed_evaluation`.
> - **What it uniquely adds:** External long-document RAG/evidence evaluation.
> - **Planned original records:** 0.
> - **Why this decision:** Benchmark purpose and contamination risk.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **MMDocRAG: Benchmarking Retrieval-Augmented Generation for Long Multimodal Documents**. |
| 2 | Where to find the paper and official release | Paper: [MMDORAG-P]. Official repository: [MMDORAG-GH]. |
| 3 | Can we get it now? | Public benchmark. |
| 4 | Exact version or revision | Official 2025 release. |
| 5 | What the license allows — and does not allow | Use source/repository terms; no training authorization assumed. |
| 6 | Where the original documents came from | 2,965 long documents across 14 types. |
| 7 | What kind of material it contains | Long multimodal documents. |
| 8 | How many documents | 2,965 documents. |
| 9 | How many pages or images | 236,230 pages. |
| 10 | How many questions | 4,055 expert QA. |
| 11 | Official train, development, and test counts | Evaluation benchmark. |
| 12 | Kinds of answers | Extractive, reasoning, and synthesis. |
| 13 | Kinds of reasoning | Retrieval, comparison, aggregation, and synthesis across long documents. |
| 14 | Where and how the evidence is spread | One-to-three evidence items/pages, depending on task. |
| 15 | What evidence labels exist and how precise they are | Expert evidence annotations at page/item level; region completeness varies. |
| 16 | How spatial coordinates are represented | No uniform fine-region coordinates. |
| 17 | Are the original PDFs or page images available? | Documents available under source terms. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Adapter possible for page/evidence evaluation. |
| 19 | Does it label the complete evidence set? | Not uniformly complete at candidate-region level. |
| 20 | Known gaps or quality concerns | Public benchmark contamination and heterogeneous document rights. |
| 21 | Could it overlap with other training or evaluation data? | Potential overlap with other long-document benchmark PDFs. |
| 22 | Real example questions from official material | Exact benchmark examples are intentionally not reproduced to preserve quarantine. |
| 23 | What this source uniquely adds | External long-document RAG/evidence evaluation. |
| 24 | How many original records this project plans to use | 0. |
| 25 | Planned augmentations and exact quotas | 0. |
| 26 | Its main job in this project — or the decision to exclude it | `sealed_evaluation`. |
| 27 | Why that decision makes sense | Benchmark purpose and contamination risk. |
| 28 | What still must be acquired or verified | Hash documents before evaluation and prevent benchmark prompts/questions from entering development logs. |

### 7.11 MMLongBench-Doc v1 and V2

> **Quick take**
> - **Main project role:** `sealed_evaluation`.
> - **What it uniquely adds:** Primary external evaluation of the final system.
> - **Planned original records:** 0.
> - **Why this decision:** Explicit user requirement and high contamination risk.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **MMLongBench-Doc** and corrected **MMLongBench-Doc-V2**. |
| 2 | Where to find the paper and official release | V1 paper/repo: [MMLBD-P], [MMLBD-GH]. V2 paper/repo/HF: [MMLBD2-P], [MMLBD2-GH], [MMLBD2-HF]. |
| 3 | Can we get it now? | Public benchmarks; completely quarantined. |
| 4 | Exact version or revision | V2 released 2026-08-04, two days before this audit cutoff. |
| 5 | What the license allows — and does not allow | Benchmark/source-document terms; no training use. |
| 6 | Where the original documents came from | Long visually rich documents across many types. |
| 7 | What kind of material it contains | Long multimodal PDFs. |
| 8 | How many documents | V1 paper 130 docs versus current repo 135 PDFs; V2 134 long documents. |
| 9 | How many pages or images | V2 average exceeds 31 pages/document; exact page total from release manifest. |
| 10 | How many questions | V1 paper 1,062 Q versus repo 1,091; V2 1,071 Q. |
| 11 | Official train, development, and test counts | External test only; V2 corrections must remain sealed. |
| 12 | Kinds of answers | 16 question types in V2; extractive, numerical, comparative, abstractive, unanswerable. |
| 13 | Kinds of reasoning | Single/multi-page and document-wide; V2 reports 33.5% unanswerable. |
| 14 | Where and how the evidence is spread | V2 supplies detailed per-page evidence; region-level compatibility depends on schema adapter. |
| 15 | What evidence labels exist and how precise they are | Page-level evidence/metadata; no universal region-box gold. |
| 16 | How spatial coordinates are represented | PDF/page coordinates from rendering. |
| 17 | Are the original PDFs or page images available? | Benchmark PDFs available. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Adapter required; do not inspect candidate oracle until frozen system is ready. |
| 19 | Does it label the complete evidence set? | Page evidence is useful but not necessarily minimal region evidence. |
| 20 | Known gaps or quality concerns | V1 contained annotation errors corrected by V2; public exposure creates contamination risk. |
| 21 | Could it overlap with other training or evaluation data? | Potential PDF overlap with MMDocIR/MMDocRAG/LongDocURL/M-LongDoc. |
| 22 | Real example questions from official material | Questions are intentionally not reproduced; quarantine forbids using examples for prompt or dataset design. |
| 23 | What this source uniquely adds | Primary external evaluation of the final system. |
| 24 | How many original records this project plans to use | 0. |
| 25 | Planned augmentations and exact quotas | 0. |
| 26 | Its main job in this project — or the decision to exclude it | `sealed_evaluation`. |
| 27 | Why that decision makes sense | Explicit user requirement and high contamination risk. |
| 28 | What still must be acquired or verified | Before final evaluation, freeze SHA-256 hashes and evaluate V2 first; report V1 only as historical compatibility. Never inspect correction logs during development. |

### 7.12 DocScope

> **Quick take**
> - **Main project role:** `sealed_evaluation`.
> - **What it uniquely adds:** Potential future high-quality evidence-set evaluation.
> - **Planned original records:** 0.
> - **Why this decision:** Promising but unreleased and unauditable.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **DocScope: …** (2026 long-document evidence/rationale benchmark). |
| 2 | Where to find the paper and official release | Paper/project: [DOCSCOPE-P], [DOCSCOPE-PROJ]. |
| 3 | Can we get it now? | **What we directly verified in the release** (`verified_release_observation`) Project announced release, but no stable downloadable dataset/license was available at cutoff. |
| 4 | Exact version or revision | 2026 preprint/project snapshot. |
| 5 | What the license allows — and does not allow | Unresolved. |
| 6 | Where the original documents came from | 262 documents. |
| 7 | What kind of material it contains | Long documents with gold page/region/fact rationales and trajectories. |
| 8 | How many documents | 262. |
| 9 | How many pages or images | Exact page count unavailable without release. |
| 10 | How many questions | 1,124 questions. |
| 11 | Official train, development, and test counts | No downloadable split manifest at cutoff. |
| 12 | Kinds of answers | Long-document QA. |
| 13 | Kinds of reasoning | Evidence trajectories and rationale selection. |
| 14 | Where and how the evidence is spread | Multiple pages/regions/facts. |
| 15 | What evidence labels exist and how precise they are | Claimed page/region/fact rationales. |
| 16 | How spatial coordinates are represented | Claimed region coordinates; unverified. |
| 17 | Are the original PDFs or page images available? | No stable public assets. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Untestable. |
| 19 | Does it label the complete evidence set? | Claimed strong evidence; cannot verify. |
| 20 | Known gaps or quality concerns | Unreleased schema/license. |
| 21 | Could it overlap with other training or evaluation data? | Unknown overlap with public PDFs. |
| 22 | Real example questions from official material | Exact official example strings were not accessible in a released data artifact. |
| 23 | What this source uniquely adds | Potential future high-quality evidence-set evaluation. |
| 24 | How many original records this project plans to use | 0. |
| 25 | Planned augmentations and exact quotas | 0. |
| 26 | Its main job in this project — or the decision to exclude it | `sealed_evaluation`. |
| 27 | Why that decision makes sense | Promising but unreleased and unauditable. |
| 28 | What still must be acquired or verified | Revisit after stable data, license, splits, and revision are public. |

### 7.13 ChartQAPro

> **Quick take**
> - **Main project role:** `sealed_evaluation`.
> - **What it uniquely adds:** External chart robustness/unanswerability evaluation.
> - **Planned original records:** 0.
> - **Why this decision:** Useful test, not training evidence.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **ChartQAPro: A More Diverse and Challenging Chart Question Answering Benchmark**. |
| 2 | Where to find the paper and official release | Project/paper: [CHARTQAPRO-PROJ], [CHARTQAPRO-P]. |
| 3 | Can we get it now? | Public evaluation benchmark. |
| 4 | Exact version or revision | Official release. |
| 5 | What the license allows — and does not allow | Use official terms; no training use in this protocol. |
| 6 | Where the original documents came from | Real-world charts and verified questions. |
| 7 | What kind of material it contains | Chart QA benchmark. |
| 8 | How many documents | 1,948 charts. |
| 9 | How many pages or images | 1,948 images. |
| 10 | How many questions | 1,948 verified questions. |
| 11 | Official train, development, and test counts | Evaluation only. |
| 12 | Kinds of answers | Mostly multiple-choice; 24.8% unanswerable. |
| 13 | Kinds of reasoning | Advanced chart reasoning, ambiguity, visual literacy. |
| 14 | Where and how the evidence is spread | One chart with multiple elements. |
| 15 | What evidence labels exist and how precise they are | Answer/question labels; complete evidence boxes are not the benchmark focus. |
| 16 | How spatial coordinates are represented | No complete region coordinates. |
| 17 | Are the original PDFs or page images available? | Images available. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Benchmark adapter only. |
| 19 | Does it label the complete evidence set? | No. |
| 20 | Known gaps or quality concerns | >70% multiple-choice and public benchmark exposure. |
| 21 | Could it overlap with other training or evaluation data? | Possible source-chart overlap with ChartQA/RefChartQA. |
| 22 | Real example questions from official material | Exact benchmark questions are not reproduced to preserve quarantine. |
| 23 | What this source uniquely adds | External chart robustness/unanswerability evaluation. |
| 24 | How many original records this project plans to use | 0. |
| 25 | Planned augmentations and exact quotas | 0. |
| 26 | Its main job in this project — or the decision to exclude it | `sealed_evaluation`. |
| 27 | Why that decision makes sense | Useful test, not training evidence. |
| 28 | What still must be acquired or verified | Hash charts against ChartQA/RefChartQA before evaluation. |

### 7.14 DocFinQA

> **Quick take**
> - **Main project role:** `calibration_or_development`.
> - **What it uniquely adds:** Calibrates retrieval over full reports.
> - **Planned original records:** 0 selector; development/calibration only.
> - **Why this decision:** Long-document context is useful, but inherited labels and overlap preclude independent training counts.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **DocFinQA: … full-document financial QA extension of FinQA**. |
| 2 | Where to find the paper and official release | Project/repository: [DOCFINQA-P], [DOCFINQA-GH]. |
| 3 | Can we get it now? | Public data/project. |
| 4 | Exact version or revision | Official release. |
| 5 | What the license allows — and does not allow | Apache-2.0 repository; underlying report/data rights apply. |
| 6 | Where the original documents came from | Full financial reports extending FinQA questions/contexts. |
| 7 | What kind of material it contains | Long financial documents represented textually/structurally. |
| 8 | How many documents | 483 documents. |
| 9 | How many pages or images | Page count from acquired reports. |
| 10 | How many questions | Conflict: approximately 7,437 versus 7,621 questions across official materials. |
| 11 | Official train, development, and test counts | Use release manifest; no selector training allocation. |
| 12 | Kinds of answers | Numerical/program-derived. |
| 13 | Kinds of reasoning | Long-context retrieval plus FinQA arithmetic. |
| 14 | Where and how the evidence is spread | Multiple pages in one report. |
| 15 | What evidence labels exist and how precise they are | Inherited support/program labels; no native boxes. |
| 16 | How spatial coordinates are represented | No uniform pixel coordinates. |
| 17 | Are the original PDFs or page images available? | Reports may be reacquired. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Useful for page-gate calibration after rendering. |
| 19 | Does it label the complete evidence set? | No complete region evidence. |
| 20 | Known gaps or quality concerns | Inherited FinQA questions are not new examples; count conflict. |
| 21 | Could it overlap with other training or evaluation data? | Direct FinQA overlap and possible report overlap with TAT-DQA/MultiHiertt. |
| 22 | Real example questions from official material | Use FinQA authentic examples; DocFinQA inherits that family and does not contribute new question identities. |
| 23 | What this source uniquely adds | Calibrates retrieval over full reports. |
| 24 | How many original records this project plans to use | 0 selector; development/calibration only. |
| 25 | Planned augmentations and exact quotas | No generative augmentation. |
| 26 | Its main job in this project — or the decision to exclude it | `calibration_or_development`. |
| 27 | Why that decision makes sense | Long-document context is useful, but inherited labels and overlap preclude independent training counts. |
| 28 | What still must be acquired or verified | Reconcile 7,437/7,621 and build FinQA question-ID mapping. |

### 7.15 “MultiDocVQA” name audit

> **Quick take**
> - **Main project role:** `exclude`.
> - **What it uniquely adds:** None.
> - **Planned original records:** 0.
> - **Why this decision:** Canonical identity is unresolved.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | No canonical dataset named exactly **MultiDocVQA** was identified in primary sources by the cutoff. |
| 2 | Where to find the paper and official release | Candidate conflations are MP-DocVQA [MPDOC-P] and M3DocVQA [M3DOC-PROJ]. |
| 3 | Can we get it now? | Unresolved name; no asset is acquired under this label. |
| 4 | Exact version or revision | Not applicable. |
| 5 | What the license allows — and does not allow | Not applicable. |
| 6 | Where the original documents came from | Unknown. |
| 7 | What kind of material it contains | Unknown. |
| 8 | How many documents | Unknown. |
| 9 | How many pages or images | Unknown. |
| 10 | How many questions | Unknown. |
| 11 | Official train, development, and test counts | Unknown. |
| 12 | Kinds of answers | Unknown. |
| 13 | Kinds of reasoning | Unknown. |
| 14 | Where and how the evidence is spread | Unknown. |
| 15 | What evidence labels exist and how precise they are | Unknown. |
| 16 | How spatial coordinates are represented | Unknown. |
| 17 | Are the original PDFs or page images available? | Unknown. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Unknown. |
| 19 | Does it label the complete evidence set? | Unknown. |
| 20 | Known gaps or quality concerns | Using an ambiguous alias would risk mixing distinct benchmarks. |
| 21 | Could it overlap with other training or evaluation data? | Likely naming collision with MP-DocVQA/M3DocVQA. |
| 22 | Real example questions from official material | No authentic examples can be supplied because no canonical release was found. |
| 23 | What this source uniquely adds | None. |
| 24 | How many original records this project plans to use | 0. |
| 25 | Planned augmentations and exact quotas | 0. |
| 26 | Its main job in this project — or the decision to exclude it | `exclude`. |
| 27 | Why that decision makes sense | Canonical identity is unresolved. |
| 28 | What still must be acquired or verified | Owner must supply a paper/release URL before the name can re-enter the audit. |

### 7.16 DocTrace (method-only discovery)

> **Quick take**
> - **Main project role:** `exclude` as a dataset; cited as related method.
> - **What it uniquely adds:** Method precedent for graph-structured evidence and SFT→RL curriculum.
> - **Planned original records:** 0.
> - **Why this decision:** It informs design but supplies no auditable corpus.


| # | Audit field | Finding |
|---:|---|---|
| 1 | Official name and other names you may see | **DocTrace: … evidence-graph SFT and RL for document reasoning**. |
| 2 | Where to find the paper and official release | Paper: [DOCTRACE-P]. |
| 3 | Can we get it now? | Paper released 2026-08-04; no stable standalone corpus release identified. |
| 4 | Exact version or revision | Initial 2026 preprint. |
| 5 | What the license allows — and does not allow | No dataset license because no corpus is adopted. |
| 6 | Where the original documents came from | Uses evidence-graph supervision across document QA sources. |
| 7 | What kind of material it contains | Method, not a canonical dataset. |
| 8 | How many documents | Not applicable. |
| 9 | How many pages or images | Not applicable. |
| 10 | How many questions | Not applicable. |
| 11 | Official train, development, and test counts | Not applicable. |
| 12 | Kinds of answers | Not applicable. |
| 13 | Kinds of reasoning | Evidence graph planning and outcome-based RL. |
| 14 | Where and how the evidence is spread | Graph over evidence items. |
| 15 | What evidence labels exist and how precise they are | Method-defined traces. |
| 16 | How spatial coordinates are represented | Not applicable. |
| 17 | Are the original PDFs or page images available? | Not applicable. |
| 18 | Can the evidence be matched to DeepSeek-OCR2 semantic units? | Not applicable. |
| 19 | Does it label the complete evidence set? | Not a data source. |
| 20 | Known gaps or quality concerns | No stable corpus/release. |
| 21 | Could it overlap with other training or evaluation data? | Potentially trains on overlapping public QA datasets; inspect if used as baseline. |
| 22 | Real example questions from official material | No dataset examples. |
| 23 | What this source uniquely adds | Method precedent for graph-structured evidence and SFT→RL curriculum. |
| 24 | How many original records this project plans to use | 0. |
| 25 | Planned augmentations and exact quotas | 0. |
| 26 | Its main job in this project — or the decision to exclude it | `exclude` as a dataset; cited as related method. |
| 27 | Why that decision makes sense | It informs design but supplies no auditable corpus. |
| 28 | What still must be acquired or verified | Revisit if authors release a stable training manifest. |


## 8. The dataset guest list — who is in, who is out, and why

**Friendly guidepost.** Here is the clean yes/no/maybe board. Every dataset gets one primary job, and every exclusion has a concrete reason rather than a vague feeling.

Every named source has one primary role. Secondary uses do not change that assignment. Counts shown in the selector columns are original training records only; planner, answerer, and RL counts are stated in the role/constraint cells and in §10.

| Dataset / release | Primary role | Minimum original selector | Full original selector | Unique contribution | Decisive constraint |
| --- | --- | ---: | ---: | --- | --- |
| Visual-CoT (DocVQA/InfographicVQA/SROIE views) | core_selector_sft | 5,500 | 27,000 | Controlled starter with pinned, source-explicit question-conditioned boxes | Level C only; license matrix required |
| Visual-CoT TextVQA view | exclude | 0 | 0 | None for the document corpus | Scene-text domain; owner-excluded |
| Visual-CoT document-view alias | exclude | 0 | 0 | None beyond explicit source views | Not an independent corpus |
| DocVQA direct | exclude | 0 | 0 | Parent provenance | Duplicate of Visual-CoT derivative; no complete boxes |
| InfographicVQA direct | exclude | 0 | 0 | Parent provenance | Derivative carries actionable boxes |
| Visual-CoT DUDE view | exclude | 0 | 0 | None | Duplicate of original DUDE |
| SROIE direct | exclude | 0 | 0 | Parent line/KIE labels | No native questions |
| TextVQA original | exclude | 0 | 0 | None for the document corpus | Scene-text domain; no native complete evidence; owner-excluded |
| TextCaps | exclude | 0 | 0 | None | Captioning, synthetic questions, image overlap |
| BoundingDocs source-distinct | auxiliary_selector_sft | 1,000 | 5,000 | Normalized forms/invoices/KIE boxes | Remove DUDE/MP/SP parents; Level C |
| BBox DocVQA v1 | exclude | 0 | 0 | Historical identity only | Superseded by SciEGQA v2 |
| SciEGQA-Train | exclude | 0 | 0 | Potential future scientific boxes | No explicit license; automatic labels unaudited |
| SciEGQA-Bench | sealed_evaluation | 0 | 0 | Human scientific multi-region/multi-page test | Benchmark quarantine |
| GroundingDocQA/M3Grounder data | exclude | 0 | 0 | Potential masks | No public data/license |
| MP-DocVQA | calibration_or_development | 0 | 0 | Page-gate and wrong-page calibration | Mostly single-answer-page; no complete regions |
| DUDE original | core_selector_sft | 4,500 | 12,500 | Real multi-page documents and native unanswerables | Audit non-extractive evidence |
| SlideVQA | sealed_evaluation | 0 | 0 | Real cross-slide evidence-page benchmark | Evaluation/internal-only license |
| QASPER | auxiliary_selector_sft | 1,000 | 1,100 | Reader-authored scientific questions and paragraph evidence | Requires lawful PDF alignment |
| FinQA | planner_or_decomposition_pretraining | 0 | 6,000 planner | Programs/operands | No native pages/boxes |
| TAT-QA | planner_or_decomposition_pretraining | 0 | 8,000 planner + 3,000 answerer | Hybrid table-text derivations | Use TAT-DQA for visual selector |
| MultiHiertt | planner_or_decomposition_pretraining | 0 | 5,500 planner | Multi-table programs | Remove 2,119 FinQA overlaps |
| ConditionalQA | planner_or_decomposition_pretraining | 0 | 2,000 planner | Condition/exception and insufficiency | Text-only |
| MESAQA | exclude | 0 | 0 | Evidence-cardinality method anchor | Unclear license; synthetic text-only |
| QuoteSum/SEMQA | answerer_sft | 0 | **1,200 answer rows** | Attributed semi-extractive synthesis | No native page grounding |
| M-LongDoc benchmark | sealed_evaluation | 0 | 0 | Long-document external test | No license; noisy evidence |
| DocDownstream10K | later_rl | 0 | 10,000 conditional | Outcome-based long-document RL | Only after rights approval |
| MultiModalQA | planner_or_decomposition_pretraining | 0 | 2,000 planner + 4,000 answerer + 4,000 RL | Multimodal composition/synthetic bundles | Remove M3DocVQA overlap |
| HotpotQA | planner_or_decomposition_pretraining | 0 | 2,000 planner | Bridge/comparison decomposition | Wikipedia/text-only |
| BREAK | planner_or_decomposition_pretraining | 0 | 1,500 planner | QDMR decomposition operators | No evidence |
| TAT-DQA | core_selector_sft | 5,000 | 12,000 | Licensed visual table-text programs/evidence | Verify genuine cross-page subset |
| PeerQA | auxiliary_selector_sft | 0 | 400 | Expert information needs and limitations | PDF alignment; noncommercial share-alike |
| RefChartQA | auxiliary_selector_sft | 1,000 | 6,000 | Chart-element grounding | AGPL/GPL conflict; legal approval |
| SPIQA | answerer_sft | 0 | **6,800 answerer + 6,000 RL** | Scientific figure/table explanations | Generated train; paper overlap |
| HiTab | planner_or_decomposition_pretraining | 0 | 5,000 planner | Hierarchical header programs | Text/table structure only |
| ChartQA | exclude | 0 | 0 | Parent provenance | RefChartQA is stronger grounded derivative |
| LongDocURL | sealed_evaluation | 0 | 0 | Long-document subtask benchmark | No selector-grade evidence |
| M3DocVQA | sealed_evaluation | 0 | 0 | Multi-document rendered visual QA | Direct MultiModalQA overlap |
| MMDocIR | sealed_evaluation | 0 | 0 | Page-retrieval external benchmark | Training mixture duplicates selected sources |
| MMDocRAG | sealed_evaluation | 0 | 0 | Long multimodal RAG external benchmark | Public benchmark contamination |
| MMLongBench-Doc V1/V2 | sealed_evaluation | 0 | 0 | Primary external long-document test | Complete quarantine; V2 primary |
| DocScope | sealed_evaluation | 0 | 0 | Potential rationale/trajectory test | Unreleased at cutoff |
| ChartQAPro | sealed_evaluation | 0 | 0 | Chart robustness and unanswerability | Evaluation benchmark; no complete boxes |
| DocFinQA | calibration_or_development | 0 | 0 | Full-report retrieval calibration | Inherited FinQA questions |
| “MultiDocVQA” alias | exclude | 0 | 0 | None | No canonical release identified |
| DocTrace | exclude | 0 | 0 | Method precedent | No standalone corpus |


## 9. The smallest experiment that can still answer the big questions

**Friendly guidepost.** This is the lean experiment: 32,000 selector-SFT records, large enough to separate coordinate generation, candidate-ID generation, and parallel set prediction without building the entire future curriculum first.

### 9.1 Decision and scale

**Our proposed plan** (`proposed_design`) **32,000 unique selector-SFT records** is the minimum publication-quality corpus. It is the smallest plan here that simultaneously contains (i) enough native/pseudo spatial single-region records to test DeepSeek-OCR2-Box versus DeepSeek-OCR2-ID, (ii) 8,000 same-page or real cross-page multi-evidence records to expose serial stopping and set-cardinality behavior, and (iii) 4,000 explicit no-evidence/adversarial records. It is not claimed to be a universal lower bound; it is the minimum supported by the power, source-diversity, and error-analysis requirements of this experiment.

The 32,000 count is **training only**. A separate 4,000-record development set and 4,000-record test set are built from untouched official development/test groups or pre-augmentation held-out visual groups (§15); they are not included below.

### 9.2 Source allocation

| Source | Source split | Verified/published availability before acquisition | Original `target_n` | Accepted augmentations | Final records | % of 32k | Supervision | Stages |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Visual-CoT DocVQA | official train only | **26,117 rows** at pinned commit | 3,000 | 2,000 | 5,000 | 15.62% | C | 0,1 |
| Visual-CoT InfographicVQA | official train only | **19,429 rows** at pinned commit | 2,000 | 1,800 | 3,800 | 11.88% | C | 0,1 |
| Visual-CoT SROIE | official train only | **2,584 rows** at pinned commit | 500 | 500 | 1,000 | 3.12% | C | 0,1 |
| DUDE | official train only | 23,728 official train questions | 4,500 | 4,200 | 8,700 | 27.19% | A/B/C by answer type | 0,1,2,5 |
| TAT-DQA | official train only | 13,251 official train questions | 5,000 | 3,700 | 8,700 | 27.19% | A/B | 0,1,2,5 |
| BoundingDocs source-distinct | official train only | 212,488 all-split QA after removing DUDE/MP/SP (`derived`); exact train-eligible count computed from release | 1,000 | 600 | 1,600 | 5.00% | C | 0,5 |
| QASPER aligned | official train only | 2,593 official train questions before PDF/evidence alignment | 1,000 | 700 | 1,700 | 5.31% | B→C after spatial alignment | 2 |
| RefChartQA | official train only | 55,789 official train rows | 1,000 | 500 | 1,500 | 4.69% | A/C | 0,1 |
| **Total** |  |  | **18,000** | **14,000** | **32,000** | **100.00%** |  |  |

For rows whose exact release line count is not yet verified, acceptance is deterministic: enumerate the pinned train JSONL in ascending `(visual_identity_group, source_question_id)` after exclusions; score eligibility; select with the stratified hash sampler in §15. If the eligible count is below `target_n`, execute the source-specific backfill in §9.7 or declare the source unable to meet quota. The total is never silently reduced or redistributed.

### 9.3 Augmentation generation and acceptance accounting

| Family | Candidates generated | Planned acceptance | Exact accepted `target_n` | Stage | Hard condition |
| --- | ---: | ---: | ---: | --- | --- |
| Meaning-preserving paraphrase | 2500 | 80% | 2,000 | Stages 0–2 | Original question retained as anchor |
| Single-page multi-evidence | 8000 | 50% | 4,000 | Stage 1 | 2–4 necessary units |
| Real same-document cross-page | 10000 | 40% | 4,000 | Stage 2 | All pages from one pre-split document |
| Missing-premise/unanswerable | 4000 | 50% | 2,000 | Stage 5 | Removed required unit/page |
| Generated query-side negative | 8000 | 25% | 2,000 | Stage 5 | Semantic property changed; fixed evidence invalid |
| **Total** | **32,500** |  | **14,000** |  |  |

The acceptance percentages are **planning assumptions**, not observed yields. Publication tables must replace them with measured yields and list every reject/ambiguous reason.

### 9.4 Source-by-augmentation matrix

| Source | Paraphrase | Same-page multi | Real cross-page | Missing premise | Query negative | Total accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Visual-CoT DocVQA | 500 | 1000 | 0 | 250 | 250 | 2000 |
| Visual-CoT InfographicVQA | 300 | 1000 | 0 | 250 | 250 | 1800 |
| Visual-CoT SROIE | 100 | 300 | 0 | 50 | 50 | 500 |
| DUDE | 400 | 600 | 2000 | 600 | 600 | 4200 |
| TAT-DQA | 400 | 800 | 1500 | 500 | 500 | 3700 |
| BoundingDocs source-distinct | 100 | 100 | 100 | 150 | 150 | 600 |
| QASPER aligned | 100 | 0 | 400 | 100 | 100 | 700 |
| RefChartQA | 100 | 200 | 0 | 100 | 100 | 500 |
| **Total** | **2,000** | **4,000** | **4,000** | **2,000** | **2,000** | **14,000** |

### 9.5 Training-stage mixture and replay

| Stage | Unique records | Exact source composition | Replay |
| --- | ---: | --- | --- |
| Stage 0 | 10,000 | DocVQA 3,000; InfoVQA 1,000; SROIE 500; DUDE 1,000; TAT-DQA 1,000; BoundingDocs 500; RefChartQA 1,000; paraphrases 2,000 | 2,000 replayed into Stage 1 |
| Stage 1 | 8,000 | InfoVQA 1,000; DUDE 1,000; TAT-DQA 2,000; same-page multi-evidence 4,000 | 2,000 Stage-0 replay presentations |
| Stage 2 | 8,000 | DUDE 2,000; TAT-DQA 1,000; QASPER 1,000; real cross-page 4,000 | 2,000 replay presentations: 1,000 each from Stages 0 and 1 |
| Stage 5 | 6,000 | DUDE 500; TAT-DQA 1,000; BoundingDocs 500; missing-premise 2,000; query negatives 2,000 | 2,000 ordinary-positive replay presentations |
| **Unique total** | **32,000** |  | **36,000 training presentations after replay** |

### 9.6 Required distributions

**Evidence cardinality**

| Stage | 0 | 1 | 2 | 3 | 4+ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 0 | 1000 | 7000 | 1500 | 500 | 0 |
| Stage 1 | 500 | 1000 | 3500 | 2500 | 500 |
| Stage 2 | 500 | 500 | 2500 | 2000 | 2500 |
| Stage 5 | 4000 | 1500 | 500 | 0 | 0 |
| **Total** | **6,000** | **10,000** | **8,000** | **5,000** | **3,000** |

**Answer form**

| Stage | Extractive | Numerical/program | List/multi-span | Yes/no | Unanswerable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 0 | 8000 | 500 | 500 | 0 | 1000 |
| Stage 1 | 3000 | 2000 | 1500 | 1000 | 500 |
| Stage 2 | 2000 | 3500 | 1500 | 500 | 500 |
| Stage 5 | 1000 | 0 | 500 | 500 | 4000 |
| **Total** | **14,000** | **6,000** | **4,000** | **2,000** | **6,000** |

**Supervision strength**

| Stage | A native spatial | B native structural | C verified pseudo |
| --- | ---: | ---: | ---: |
| Stage 0 | 2000 | 3000 | 5000 |
| Stage 1 | 1500 | 2500 | 4000 |
| Stage 2 | 1000 | 2000 | 5000 |
| Stage 5 | 500 | 500 | 5000 |
| **Total** | **5,000** | **8,000** | **19,000** |

**Reasoning operations**

| Stage | Exact counts |
| --- | --- |
| Stage 0 | Direct lookup 7,000; entity resolution 1,000; header dependency 1,000; missing-premise 1,000 |
| Stage 1 | Header/cell 2,500; comparison 1,500; arithmetic 1,000; condition/exception 1,000; chart/caption 1,000; claim/qualification 1,000 |
| Stage 2 | Arithmetic 2,000; comparison 1,500; aggregation 1,000; temporal 1,000; reconciliation 1,000; definition/application 500; claim/qualification 500; entity resolution 500 |
| Stage 5 | Missing-premise 4,000; reconciliation 1,000; direct hard-negative 1,000 |

### 9.7 Deterministic backfill

1. **Within-source backfill:** take the next hash-ranked eligible parent in the same source, answer form, evidence cardinality, and reasoning stratum.
2. **Topology-preserving source backfill:** QASPER shortage → PeerQA, then DUDE/TAT-DQA real cross-page; BoundingDocs shortage → Visual-CoT DocVQA, then TAT-DQA; RefChartQA legal denial → TAT-DQA and Visual-CoT InfographicVQA with the same cardinality/reasoning mix.
3. **Family-preserving synthetic backfill:** generate another batch from unused parents in the same source/stratum. A parent can yield at most one accepted record per augmentation family in the minimum plan.
4. **Failure declaration:** if the replacement pool cannot preserve the stage, topology, answer form, and supervision totals, mark the plan infeasible. Do not substitute arbitrary single-hop records.


## 10. The full recommended training plan

**Friendly guidepost.** This is the full build: 128,000 selector-SFT records plus separate planner, answerer, and possible later-RL pools. The tables spell out where every record comes from and where it goes.

### 10.1 Scope

**Our proposed plan** (`proposed_design`) The preferred selector corpus contains **128,000 unique SFT records: 64,000 original/aligned records plus 64,000 verified derivatives**. Separate planner, answerer, and conditional RL pools bring the future program to 196,000 unique records, but those pools are not allowed to masquerade as selector supervision.

The selector count excludes an 8,000-record internal development set and 16,000-record internal test set (§15), plus every sealed benchmark.

### 10.2 Selector source allocation

| Source | Source split | Verified/published availability | Original `target_n` | Accepted augmentations | Final selector records | % of 128k | Supervision |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Visual-CoT DocVQA | train / custom train | **26,117 pinned rows** | **16,000** | 9,600 | **25,600** | **20.00%** | C |
| Visual-CoT InfographicVQA | train / custom train | **19,429 pinned rows** | 9,000 | 7,500 | 16,500 | 12.89% | C |
| Visual-CoT SROIE | train / custom train | **2,584 pinned rows** | **2,000** | 2,100 | **4,100** | **3.20%** | C |
| DUDE | train / custom train | 23,728 train questions | 12,500 | 13,400 | 25,900 | **20.24%** | A/B/C |
| TAT-DQA | train / custom train | 13,251 train questions | 12,000 | 15,100 | 27,100 | 21.17% | A/B |
| BoundingDocs source-distinct | train / custom train | 212,488 all-split source-distinct QA; train eligible count derived at acquisition | 5,000 | 3,300 | 8,300 | **6.49%** | C |
| QASPER aligned | train / custom train | 2,593 train questions before alignment | 1,100 | 6,200 | 7,300 | 5.70% | B/C |
| PeerQA aligned | train / custom train | 579 questions over 208 papers; custom paper-disjoint split | 400 | 2,200 | 2,600 | 2.03% | B/C |
| RefChartQA | train / custom train | 55,789 train rows | 6,000 | 4,600 | 10,600 | 8.28% | A/C |
| **Total** |  |  | **64,000** | **64,000** | **128,000** | **100.00%** |  |

### 10.3 Full augmentation accounting

| Family | Candidates generated | Planned acceptance | Exact accepted `target_n` |
| --- | ---: | ---: | ---: |
| Meaning-preserving paraphrase | 10000 | 80% | 8,000 |
| Single-page multi-evidence | 24000 | 50% | 12,000 |
| Question-pair composition | 15000 | 40% | 6,000 |
| Real same-document cross-page | 25000 | 40% | 10,000 |
| Synthetic cross-document bundle | 10000 | 40% | 4,000 |
| Semi-extractive synthesis | 16000 | 50% | 8,000 |
| Fully abstractive multi-evidence | 15000 | 40% | 6,000 |
| Missing-premise/unanswerable | 10000 | 50% | 5,000 |
| Generated query-side negative | 20000 | 25% | 5,000 |
| **Total** | **145,000** |  | **64,000** |

| Source | Paraphrase | Same-page multi | Question composition | Real cross-page | Synthetic bundle | Semi-extractive | Abstractive | Missing premise | Query negative | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Visual-CoT DocVQA | 2,100 | 3,000 | 900 | 0 | 500 | 900 | 400 | 900 | 900 | 9,600 |
| Visual-CoT InfographicVQA | 1,200 | 2,500 | 900 | 0 | 500 | 800 | 400 | 600 | 600 | 7,500 |
| Visual-CoT SROIE | 400 | 800 | 300 | 0 | 0 | 200 | 0 | 200 | 200 | 2,100 |
| DUDE | 1,400 | 1,400 | 900 | 4,000 | 500 | 1,800 | 1,600 | 900 | 900 | 13,400 |
| TAT-DQA | 1,400 | 2,500 | 1,600 | 3,500 | 800 | 1,800 | 1,500 | 1,000 | 1,000 | 15,100 |
| BoundingDocs source-distinct | 500 | 500 | 300 | 500 | 300 | 300 | 100 | 400 | 400 | 3,300 |
| QASPER aligned | 500 | 0 | 400 | 1,500 | 500 | 1,300 | 1,200 | 400 | 400 | 6,200 |
| PeerQA aligned | 100 | 0 | 100 | 500 | 200 | 600 | 500 | 100 | 100 | 2,200 |
| RefChartQA | 400 | 1,300 | 600 | 0 | 700 | 300 | 300 | 500 | 500 | 4,600 |
| **Total** | **8,000** | **12,000** | **6,000** | **10,000** | **4,000** | **8,000** | **6,000** | **5,000** | **5,000** | **64,000** |

### 10.4 Training-stage allocation

| Stage | Unique records | Exact source composition | Replay |
| --- | ---: | --- | --- |
| Stage 0 | 32,000 | **DocVQA 11,000**; InfoVQA 4,000; **SROIE 2,000**; DUDE 1,000; TAT-DQA 2,000; BoundingDocs 3,000; RefChartQA 1,000; paraphrases 8,000 | 4,000 Stage-0 records replayed in Stage 1 |
| Stage 1 | 32,000 | DocVQA 2,000; InfoVQA 3,000; DUDE 2,000; TAT-DQA 4,000; BoundingDocs 1,000; RefChartQA 4,000; same-page multi 12,000; composition 4,000 | 4,000 Stage-0 replay |
| Stage 2 | 28,000 | DocVQA 1,000; InfoVQA 1,000; DUDE 4,500; TAT-DQA 5,000; BoundingDocs 1,000; QASPER 500; RefChartQA 1,000; real cross-page 10,000; synthetic bundle 4,000 | 4,000 replay: 2,000 each from Stages 0 and 1 |
| Stage 3 | 12,000 | DUDE 2,000; QASPER 600; PeerQA 400; DocVQA 500; InfoVQA 500; semi-extractive 8,000 | 2,000 replay: 1,000 each from Stages 1 and 2 |
| Stage 4 | 8,000 | DUDE 1,000; TAT-DQA 500; DocVQA 500; abstractive 6,000 | 2,000 replay from Stages 1–3 |
| Stage 5 | 16,000 | DocVQA 1,000; InfoVQA 500; DUDE 2,000; TAT-DQA 500; composition 2,000; missing-premise 5,000; query negatives 5,000 | 8,000 ordinary-positive replay from Stages 0–4 |
| **Unique total** | **128,000** |  | **148,000 training presentations after replay** |

### 10.5 Required distributions

**Evidence cardinality**

| Stage | 0 | 1 | 2 | 3 | 4+ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 0 | 2000 | 24000 | 6000 | 0 | 0 |
| Stage 1 | 1000 | 3000 | 16000 | 8000 | 4000 |
| Stage 2 | 1000 | 1000 | 10000 | 8000 | 8000 |
| Stage 3 | 500 | 500 | 4000 | 3000 | 4000 |
| Stage 4 | 500 | 500 | 2000 | 2000 | 3000 |
| Stage 5 | 12000 | 2000 | 1000 | 500 | 500 |
| **Total** | **17,000** | **31,000** | **39,000** | **21,500** | **19,500** |

**Answer form**

| Stage | Extractive | Numerical/program | List/multi-span | Yes/no | Semi-extractive | Abstractive | Unanswerable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 0 | 22000 | 4000 | 2000 | 2000 | 0 | 0 | 2000 |
| Stage 1 | 10000 | 8000 | 6000 | 2000 | 2000 | 3000 | 1000 |
| Stage 2 | 4000 | 10000 | 4000 | 1000 | 3000 | 5000 | 1000 |
| Stage 3 | 1000 | 1000 | 1000 | 500 | 7000 | 1000 | 500 |
| Stage 4 | 500 | 500 | 500 | 500 | 1000 | 4500 | 500 |
| Stage 5 | 2000 | 500 | 500 | 500 | 500 | 0 | 12000 |
| **Total** | **39,500** | **24,000** | **14,000** | **6,500** | **13,500** | **13,500** | **17,000** |

**Supervision strength**

| Stage | A native spatial | B native structural | C verified pseudo |
| --- | ---: | ---: | ---: |
| Stage 0 | 8000 | 4000 | 20000 |
| Stage 1 | 6000 | 8000 | 18000 |
| Stage 2 | 4000 | 10000 | 14000 |
| Stage 3 | 0 | 4000 | 8000 |
| Stage 4 | 0 | 2000 | 6000 |
| Stage 5 | 2000 | 2000 | 12000 |
| **Total** | **20,000** | **30,000** | **78,000** |

**Reasoning operations**

| Stage | Exact counts |
| --- | --- |
| Stage 0 | Direct 20,000; entity 3,000; headers 4,000; chart 2,000; missing-premise 3,000 |
| Stage 1 | Headers 8,000; comparison 6,000; arithmetic 4,000; chart 4,000; condition/exception 3,000; claim/qualification 3,000; entity 2,000; reconciliation 2,000 |
| Stage 2 | Arithmetic 7,000; comparison 4,000; aggregation 4,000; temporal 3,000; headers 2,000; reconciliation 3,000; definition/application 2,000; claim 1,000; document-wide count 1,000; entity 1,000 |
| Stage 3 | Claim/qualification 3,000; reconciliation 2,000; causal 2,000; comparison 1,500; definition/application 1,000; document synthesis 1,500; direct 1,000 |
| Stage 4 | Causal 3,000; reconciliation 1,500; comparison 1,000; claim/qualification 1,000; definition/application 500; document synthesis 1,000 |
| Stage 5 | Missing-premise 12,000; reconciliation 2,000; direct hard-negative 1,000; comparison-direction 1,000 |

### 10.6 Non-selector pools

| Pool | Target records | Exact composition | Use restriction |
| --- | ---: | --- | --- |
| Planner/decomposition pretraining | 32,000 | FinQA 6,000; TAT-QA 8,000; MultiHiertt 5,500 overlap-clean; ConditionalQA 2,000; HiTab 5,000; MultiModalQA 2,000; HotpotQA 2,000; BREAK 1,500 | No selector labels; source IDs disjoint from answerer/RL pools |
| Answerer SFT | 16,000 | **QuoteSum 1,200** answer rows; QASPER 1,000; **SPIQA 6,800**; TAT-QA 3,000; MultiModalQA 4,000 | Disjoint questions/source groups from selector/planner where applicable |
| Conditional later RL | 20,000 | DocDownstream10K 10,000 pending license; SPIQA 6,000; MultiModalQA 4,000 | Outcome reward only after selector SFT; weak labels never back-propagated as gold evidence |
| **Total future unique pool** | **196,000** | 128,000 selector + 32,000 planner + 16,000 answerer + 20,000 RL | Counts exclude all sealed evaluation |

If DocDownstream10K rights are denied, its 10,000 RL quota is backfilled in this fixed order: SPIQA +4,000; MultiModalQA +3,000; ConditionalQA +2,000; held-out QASPER/TAT-QA answerer records +1,000. The selector corpus is unchanged.

### 10.7 Full-plan backfill and source ceilings

- **QASPER:** shortage after lawful PDF alignment → PeerQA up to its remaining paper-disjoint ceiling, then DUDE/TAT-DQA real-cross-page rows.
- **BoundingDocs:** shortage after parent-license/source-distinct filters → Visual-CoT DocVQA, then TAT-DQA, matching stage/cardinality/reasoning.
- **RefChartQA:** legal denial → TAT-DQA chart/table rows and Visual-CoT InfographicVQA; the paper must report that chart-element grounding coverage was reduced.
- **SciEGQA future addendum:** after explicit license and audit, it may replace at most 4,000 Visual-CoT/BoundingDocs Level-C records. It does not increase the 128,000 total and cannot touch internal or external evaluation questions.
- **No source may contribute more than 20 near-template questions from one document to one epoch.** Excess questions remain eligible for later epochs only through document-balanced sampling.


## 11. How we create useful new training examples

**Friendly guidepost.** Augmentation should create useful variety, not synthetic shortcuts. Each family starts from an authoritative parent example and must prove that the answer and required evidence still mean what they meant before.

All generated records retain their original question as the authoritative anchor and all parent IDs. Generation occurs **after** split assignment and candidate freezing. No generator sees sealed evaluation data.

### 11.1 Meaning-preserving positive rewrites

**Generation contract.** Input: authoritative original question, canonical answer, complete evidence set, answer type, semantic slots. Generate one paraphrase without showing the answer text as an instruction. Preserve entity, time, polarity, scope, comparison direction, quantity/metric, and expected answer type. Store `question_anchor` as the original.

**Acceptance contract.** Reject if any slot changes; if original and rewrite do not induce the same accepted evidence set under two independent verifiers; if lexical overlap is >0.92 token Jaccard (trivial) or semantic similarity <0.82 (intent drift); or if answerability changes. Minimum 2,000; full 8,000.

### 11.2 Single-page multi-evidence construction

**Generation contract.** Start only from a Level-A/B or verified Level-C parent. A generator sees the whole page and candidate graph, chooses a second/third unit related by header/value, claim/caveat, rule/exception, chart/legend, comparison, definition/application, or paragraph/footnote, then writes one natural question and answer requiring all units.

**Acceptance contract.** Enumerate all proper evidence subsets for cardinality ≤4. Reject if any subset entails/answers the target, if units are unrelated, if the second unit is decorative, or if the answer uses outside knowledge. Minimum 4,000; full 12,000.

### 11.3 Cross-page same-document construction

**Generation contract.** Split documents before generation. Select complementary pages in one real document by structural links, repeated entities, section references, table continuation, definition→result, method→outcome, baseline→later result, policy→exception, or finding→limitation.

**Acceptance contract.** Require all parent page hashes in one split and the final evidence set to include at least two pages. Reject if one page alone answers, if pages merely duplicate a fact, or if page order/identity is exposed by synthetic wording. Minimum 4,000; full 10,000.

### 11.4 Synthetic cross-document page bundles

**Generation contract.** Choose pages from different documents only after creating an explicit bundle record. Pages must share a coherent comparison frame, topic/entity/metric, or multi-source synthesis goal. Preserve every source document ID and set `synthetic_bundle=true`.

**Acceptance contract.** Never label as same-document. Require source compatibility and a question that names/implicitly defines the comparison frame. Evaluate separately. Minimum 0; full 4,000.

### 11.5 Question-pair composition

**Generation contract.** Eligible parent pairs must have compatible entities/metrics/time and non-overlapping necessary evidence. Supported transforms: comparison, difference, ratio, aggregation, temporal ordering, conditional decision, reconciliation, multi-source summary. Store both parent questions/answers/evidence, transform, derived program, answer, and validation.

**Acceptance contract.** Reject concatenated independent lookups, ambiguous units, incompatible denominators, non-deterministic arithmetic, or answer leakage. Minimum 0; full 6,000.

### 11.6 Semi-extractive synthesis

**Generation contract.** Use QuoteSum/SEMQA as the answer-format anchor. Generate an answer containing attributed factual spans from each evidence role plus connective language. The selector target remains the evidence set, while the answerer target preserves citation/attribution markers.

**Acceptance contract.** Every factual clause must align to a selected evidence unit; connective text may not introduce new facts. Reject unsupported claims or quotation boundaries that do not match source text. Full 8,000.

### 11.7 Fully abstractive multi-evidence

**Generation contract.** Create explanation, comparison, limitation, cause, implication, or document-level synthesis only from Level-A/B or verified Level-C evidence. Build a claim-evidence matrix before drafting the answer.

**Acceptance contract.** Reject generic summaries, unverifiable claims, evidence sets with unused units, or answers answerable from a proper subset. Full 6,000; Stage 4 only.

### 11.8 Missing-premise and unanswerable

**Generation contract.** Derive from a verified positive by removing one required page/unit, altering an entity/year/metric/property, withholding an exception, or retaining only one comparison operand. Distinguish `missing_required_evidence`, `unsupported_property`, and native `unanswerable`.

**Acceptance contract.** Search the entire page/document and candidate universe. If the answer exists elsewhere, the record is not a no-evidence example; route to positive or `ignore`. Minimum 2,000; full 5,000.

### 11.9 Spatial transformations

**Generation contract.** Allowed: deterministic rerendering at 150/200/300 dpi; isotropic scale; padding; color/grayscale conversion; bounded ±1.5° rotation only when all boxes remain valid; JPEG/PNG encoding; coordinate normalization; snapping boxes to semantic units with logged IoU/text coverage.

**Acceptance contract.** Transform the image and every box through the same affine matrix. Reject any crop that removes evidence, non-uniform warp that changes table geometry, OCR reflow, or box clipping >1% area. Spatial variants are presentations of the same record, not new semantic examples.

### 11.10 Generator independence and prompt control

- Use at least two generator families across the corpus; no generator may also be the sole verifier.
- Record model name, immutable revision, decoding parameters, system/user prompt hashes, and exact input candidate IDs.
- The selector backbone (Qwen3-VL-Reranker-2B for MiniVGent) may participate as one diagnostic judge but cannot be the deciding verifier.
- Prompt templates are selected on non-evaluation pilot records and frozen before any internal test or external benchmark is opened.
- Generated wording must not mention “region,” “box,” “page A/page B,” “first question,” “second fact,” or any construction metadata unless such language is natural in the source task.


## 12. How we build synthetic multi-step questions without fooling ourselves

**Friendly guidepost.** A question is not truly multi-step just because it mentions two facts. The construction and verification rules here require each evidence role to do real work.

### 12.1 Verification state machine

Each candidate receives exactly one terminal state:

- `pass`: every hard gate passes and no unresolved verifier disagreement remains;
- `reject`: a hard gate fails or a human reviewer confirms invalidity;
- `ambiguous_ignore`: evidence necessity, naturalness, or answerability cannot be established. It is excluded from positive and negative losses but retained in the audit log.

| Stage | Explicit criterion | Disposition |
| --- | --- | --- |
| 1. Schema/format | JSON Schema validates; all parent IDs, hashes, evidence roles, and coordinate bounds present | Hard reject on failure |
| 2. Full-set answerability | Two independent answerers each achieve exact/normalized match for categorical answers or semantic score ≥0.85 for free-form; program executes exactly when present | Reject if either independent path fails; ambiguous if one passes |
| 3. Proper-subset insufficiency | For |E|≤4 test every proper nonempty subset; for |E|>4 test every leave-one-out set plus all singleton and role-group subsets. Entailment ≤0.20 and answer score drop ≥0.25 from full set | Reject if any subset remains sufficient; ambiguous when model prior gives answer without textual entailment |
| 4. Q/A/program/evidence consistency | Program result equals normalized answer; every operand resolves to one evidence role; answer type matches schema | Hard reject |
| 5. Selector-input leakage | Question/candidate metadata contains no gold answer, gold label, evidence-role name, verifier output, or parent answer alias unavailable at inference | Hard reject |
| 6. Naturalness | Two independent judges: mean ≥4/5 for fluency and plausibility; synthetic-construction detectability ≤2/5 | Ambiguous below threshold; human escalation |
| 7. Document/domain coherence | Entities, units, dates, and discourse relations are compatible with source document/bundle | Reject contradictions; ambiguous if relation is merely plausible |
| 8. Duplicate/near-duplicate | No exact duplicate; normalized question cosine <0.94 against accepted derivatives unless answer/evidence topology differs; MinHash Jaccard <0.90 | Keep one deterministic canonical record |
| 9. Contradiction | NLI contradiction probability <0.10 for every question premise and answer claim against complete evidence | Reject ≥0.20; human review 0.10–0.20 |
| 10. Independent final verification | A verifier model family not used for generation returns `answerable_full=true`, `all_required=true`, and exact evidence IDs/roles | Reject disagreement; never let generator self-certify |
| 11. Human audit | Stratified audit by source, family, cardinality, answer type, and verifier margin | Required before release |

### 12.2 Necessity test

Let the gold evidence set be `E={e1,…,ek}`. The record is selector-SFT eligible only if:

1. the answer is entailed by `E` and the fixed downstream answerer can recover it from `E` alone;
2. for every tested proper subset `S⊂E`, the answer is not entailed by `S`; and
3. each `ei` has an assigned role (`operand`, `header`, `definition`, `exception`, `qualifier`, `legend`, `comparison_target`, `supporting_claim`, or `context_anchor`) used by the reasoning program or claim-evidence graph.

For arithmetic, the program provides a deterministic necessity witness. For free-form synthesis, each factual clause must have at least one evidence role and at least one role must disappear under each leave-one-out test. If world knowledge allows a model to guess the answer but the subset does not entail it, the subset remains insufficient; if both judges cannot distinguish guessing from entailment, the record is `ambiguous_ignore`.

### 12.3 Human review sample and escalation

**Our proposed plan** (`proposed_design`) Review **200 accepted and 100 rejected/ambiguous candidates per augmentation family** after the first verifier run, stratified across sources and cardinalities. Use Wilson 95% intervals. Expand to 500 accepted candidates for any family whose estimated invalid-positive rate exceeds 2%, whose upper 95% bound exceeds 3%, or whose source-specific stratum has fewer than 30 audited records. Stop generation for a family if the confirmed invalid-positive rate exceeds 5% after adjudication.

Two reviewers independently label answerability, sufficiency, necessity, naturalness, and evidence boundaries. Cohen’s κ must be ≥0.75 for binary validity and Krippendorff’s α ≥0.70 for evidence-role/cardinality judgments; otherwise revise guidelines and re-review the entire pilot sample. Disagreements go to a third adjudicator.

### 12.4 Release accounting

For every family and source, publish: generated count, schema rejects, full-set failures, subset-necessity failures, naturalness failures, duplicate removals, contradictions, verifier disagreements, human reversals, accepted count, and `ambiguous_ignore` count. The target quota is filled only by generating additional unused parents; rejected candidates are never relabeled to hit a number.


## 13. Tricky wrong answers — and how we avoid labeling good evidence as bad

**Friendly guidepost.** Hard negatives are difficult but genuinely wrong choices. False negatives are relevant choices mislabeled as wrong. This section is designed to create more of the first without quietly manufacturing the second.

### 13.1 DocReRank as an anchor, not a template

**What the official source says** (`primary_source_fact`) DocReRank generates 12 perturbed queries per page, modifies domain-relevant properties such as year, company, numeric value, metric, and segment, verifies unanswerability with two prompts to the same VLM family, retains three negatives per positive, and reports a 20k-page/60k-negative training construction with True/False cross-entropy and a 3:1 positive-class weight. The paper does **not** report a complete raw-to-retained verifier count. [DOCRE-P] [DOCRE-PROJ]

**Our proposed plan** (`proposed_design`) This project adopts controlled semantic perturbations and retained ordinary mined negatives, but modifies the verifier in four decisive ways: independent generator/verifier families; whole-document rather than same-page answerability search; an explicit `ambiguous_ignore` state; and human estimation of false-negative rate.

### 13.2 Negative families

| Family | Construction | False-negative safeguard |
| --- | --- | --- |
| Same-page semantically similar wrong region | Same topic/entity but no required answer role | Document-wide answerability check; exclude candidates that supply any gold role |
| Same row / wrong column | Table cell shares row but wrong metric/year | Verify header path and program operand mismatch |
| Same column / wrong row | Shares metric/header but wrong entity/period | Verify row key mismatch |
| Repeated entity/date/metric/value | Lexically identical distractor occurrence | Check whether occurrence is alternative sufficient evidence; if yes mark positive/ignore |
| Parent/child/sibling structural distractor | Heading, neighboring paragraph, sibling table block | Do not mark a parent heading negative when it is required context |
| Same-document wrong page | High retrieval score, no gold role | Search entire page and annotation alternatives |
| Cross-document same-format page | Template-similar invoice/report page | Hash/provenance preserved; no cross-source answer |
| Retrieval-mined near miss | Top non-gold from BM25/text/multimodal/reranker union | Two verifier judgments; high-rank does not imply negative |
| Minimally altered query | Change one semantic property | Record property and prove fixed evidence cannot answer |
| Missing premise | Remove one required item/page | Verify answer absent elsewhere in document |
| Adversarial relation/direction | Change polarity, comparison direction, quantity, segment, or metric | Program/entailment contradiction and document-wide search |

### 13.3 Query-side negative protocol

1. Retain the original positive question and fixed evidence set.
2. Change exactly one logged property from `{entity,time,polarity,relation,quantity,metric,segment,comparison_direction,scope}`; multi-property adversaries are a separate stress-test tag.
3. Verify that the changed query is not answerable from the fixed evidence.
4. Search all candidates/pages in the source document. If answerable elsewhere, it becomes a valid **different positive** only after new evidence annotation; it is not a negative for the document.
5. Obtain two independent answerability judgments plus deterministic program/NLI checks where available.
6. Disagreement → `ambiguous_ignore`.
7. Human-audit at least 200 accepted generated negatives/source family and report false-negative rate with a 95% Wilson interval.

### 13.4 Mined-negative control

Every experiment containing generated negatives also retains a matched control of conventional retrieval-mined negatives. Candidate discovery is the union of BM25 OCR, Qwen text embeddings, ColQwen2 visual late interaction, and a pretrained multimodal reranker. Final labels use the unmodified original question and the independent evidence verifier; mining models never define truth.

### 13.5 In-batch negatives

In-batch examples are used as negatives only when their `visual_identity_group` and document provenance differ and the question is verified not answerable from the other item. Unchecked in-batch pairing is prohibited for same-document or templated financial/form sources.

### 13.6 False-negative release gate

**Our proposed plan** (`proposed_design`) The combined negative pool advances only if the human-audited false-negative point estimate is ≤1% and the upper 95% Wilson bound is ≤2%. Source/family strata above the bound are removed or relabeled; they are not averaged away by cleaner strata.


## 14. One record format that works for every experimental arm

**Friendly guidepost.** One schema keeps all three main arms — plus the diagnostic arm — on the same playing field. It also preserves the trail back to the original source and every augmentation or verification decision.

### 14.1 Canonical JSON object

```json
{
  "schema_version": "evidence_loc_v1.0",
  "record_id": "string",
  "source_dataset": "string",
  "source_version": "string",
  "source_split": "train|dev|test|other",
  "source_question_id": "string|null",
  "source_document_id": "string",
  "source_page_ids": ["string"],
  "source_pdf_hash": "sha256|null",
  "source_page_hashes": ["sha256"],
  "visual_identity_group": "string",
  "split_group_id": "string",
  "parent_record_ids": ["string"],
  "augmentation_type": "native|paraphrase|same_page_multi|real_cross_page|synthetic_bundle_cross_page|question_composition|semi_extractive|abstractive|missing_premise|query_negative|spatial_view",
  "synthetic_bundle": false,
  "question_anchor": "string",
  "question": "string",
  "answer": "string|number|array|null",
  "answer_aliases": ["string"],
  "answer_type": "extractive|multi_span|yes_no|numerical_program|semi_extractive|abstractive|unanswerable",
  "reasoning_type": ["enum"],
  "reasoning_program": "object|null",
  "subgoals": ["object"],
  "evidence_roles": [{"role_id":"string","role_type":"enum","candidate_ids":["string"],"claim_ids":["string"]}],
  "evidence_pages": ["string"],
  "evidence_spans": ["object"],
  "gold_boxes": [{"page_id":"string","x1":0.0,"y1":0.0,"x2":1.0,"y2":1.0,"source":"native|derived|pseudo"}],
  "box_coordinate_system": "normalized_xyxy_0_1",
  "candidate_revision": "string",
  "candidate_universe": [{"candidate_id":"string","page_id":"string","candidate_type":"text|table|figure|caption|header|fallback","box":[0,0,1,1],"text":"string|null"}],
  "gold_candidate_ids": ["string"],
  "gold_evidence_groups": [["candidate_id"]],
  "evidence_group_logic": "all|required_alternative_groups",
  "ignored_candidate_ids": ["string"],
  "evidence_cardinality": 0,
  "evidence_topology": "zero|one_region|same_page_multi|real_same_document_cross_page|multi_document|synthetic_bundle_cross_page|global",
  "supervision_strength": "A|B|C|D",
  "answerable": true,
  "negative_type": "enum|null",
  "changed_semantic_property": "enum|null",
  "proper_subset_tests": ["object"],
  "oracle_coverage": {"atomic":1.0,"strict_complete":true},
  "generator_model": "model@revision|null",
  "generator_prompt_hash": "sha256|null",
  "verifier_models": ["model@revision"],
  "verifier_prompt_hashes": ["sha256"],
  "verification_results": ["object"],
  "verification_release": "string",
  "human_review_status": "not_sampled|pass|reject|ambiguous|adjudicated",
  "license": "string",
  "license_restrictions": ["string"],
  "data_use_scope": "selector_sft|answerer_sft|planner|rl|development|evaluation",
  "provenance": ["uri"],
  "split": "train|dev|test|sealed_eval",
  "selection_budget": {"max_candidates":2048,"max_area_ratio":1.0,"max_visual_tokens":null}
}
```

### 14.2 Invariants

- `question_anchor` is immutable and equals the original human/source question for all derivatives.
- `gold_boxes` and `gold_candidate_ids` express the same evidence under the frozen `candidate_revision`. If a box cannot map to a candidate, strict oracle coverage fails; the record is excluded from Arms B/C unless an approved fallback candidate is added for **all** arms.
- Alternative equivalent evidence is represented by `gold_evidence_groups`; it is not forced into a single arbitrary gold set.
- `ignored_candidate_ids` contains plausible context, annotation-ambiguous units, and unverified neighboring regions. It is excluded from negative loss.
- Level-D records may set evidence fields null/empty and cannot have `data_use_scope=selector_sft`.
- Every augmented record links to every parent; provenance is a directed acyclic graph.

### 14.3 Arm projections

- **Arm A:** consumes image bundle + question and trains on normalized `gold_boxes`; alternative box groups are serialized as equivalent target orderings. Empty set is `[]`.
- **Arm B / Bq:** consumes identical image bundle, frozen candidate universe, and question; target is a canonical sorted list of `gold_candidate_ids`, with alternative groups represented by multiple valid target sequences during loss/evaluation.
- **Arm C:** consumes the same universe; target is a multi-hot vector plus optional cardinality/sufficiency labels. `ignored_candidate_ids` receive zero loss, not negative labels.


## 15. Keeping train and test clean, traceable, and duplicate-free

**Friendly guidepost.** This section keeps the evaluation honest. Documents are grouped before splitting, every derivative follows its parent, and visual or textual duplicates are detected even when filenames change.

### 15.1 Split order

1. Acquire and normalize only metadata/annotations needed to construct identity groups.
2. Compute `visual_identity_group` **before** any augmentation from canonical PDF/page hashes, parent IDs, and cross-dataset duplicate links.
3. Preserve official document-disjoint splits. Only official train data can enter training; official dev/test never backfill train.
4. For sources without a usable official split, assign the entire identity group using seed `evidence-loc-20260806-v1`:

```text
u = uint64(first_16_hex(sha256(seed + "|" + split_group_id))) / 2^64
train if u < 0.90
internal_dev if 0.90 <= u < 0.95
internal_test otherwise
```

5. Generate every derivative **after** assignment. All parents and descendants remain in the same split.

### 15.2 Identity and duplicate signals

- exact PDF/file SHA-256;
- canonical rendered-page SHA-256 after deterministic rasterization;
- pHash and dHash at multiple resolutions;
- OCR/text normalization plus MinHash/SimHash and character 5-gram Jaccard;
- document title/URL/DOI/arXiv/S2ORC provenance;
- question and answer exact-normalized hashes;
- semantic question embeddings and answer embeddings;
- table-data fingerprints and chart perceptual/data hashes.

Pages copied across datasets are one visual identity even when filenames, DPI, crop, or compression differ. A candidate duplicate enters manual review when pHash distance ≤6, OCR MinHash Jaccard ≥0.85, or normalized question cosine ≥0.94.

### 15.3 Internal evaluation sizes

- **Minimum plan:** 4,000 development and 4,000 test records, each exactly 1,000 single-region, 1,000 same-page multi-evidence, 1,000 real cross-page, and 1,000 no-evidence.
- **Full plan:** 8,000 development and 16,000 test records. The 16,000 test set contains 4,000 native/single-region, 4,000 same-page multi, 4,000 real cross-page, 2,000 unanswerable/missing-premise, 1,000 semi-/fully-abstractive, and 1,000 synthetic-bundle records reported separately.

Internal sets use untouched source questions where possible; synthetic test records are generated with prompts/models disjoint from training generation and receive 100% human validation.

### 15.4 Cross-dataset contamination matrix

| Risk pair/group | Detection | Disposition |
| --- | --- | --- |
| Visual-CoT ↔ DocVQA/InfographicVQA/SROIE/TextVQA/TextCaps/DUDE | Exact parent question/image IDs plus page hashes | Count retained document derivatives once; TextVQA/TextCaps are excluded entirely |
| DUDE ↔ Visual-CoT/BoundingDocs/MMDocIR | Document/page hashes + source IDs | Original DUDE is canonical; remove derivatives/mixtures |
| DocVQA/MP-DocVQA ↔ BoundingDocs/MMDocIR | Image/PDF hashes and OCR MinHash | Exclude MP/SP parents from BoundingDocs sampling |
| TAT-QA ↔ TAT-DQA | Question IDs, answer/program hashes, report IDs | Selector uses TAT-DQA; planner/answerer pools objective-disjoint |
| FinQA ↔ MultiHiertt/DocFinQA | Known 2,119 question mapping + report/question hashes | Remove overlap from MultiHiertt; DocFinQA development only |
| ChartQA ↔ RefChartQA/ChartQAPro | Chart perceptual hashes + question IDs | RefChartQA canonical train source; other benchmarks quarantined |
| MultiModalQA ↔ M3DocVQA | Question IDs/supporting contexts | Remove all M3DocVQA evaluation questions from all train pools |
| Scientific papers across QASPER/PeerQA/SPIQA/SciEGQA/DocScope | DOI/arXiv/S2ORC IDs + PDF SHA-256 | Paper-disjoint across train/internal/external evaluation |
| Long-document benchmarks | PDF/page hashes, OCR MinHash, title/URL provenance | Any overlap with sealed evaluation removes the training document |
| Wikipedia-derived sources | Page revision/title/entity IDs + text MinHash | Source-title disjoint for train/evaluation |

### 15.5 Template/entity leakage

Cluster questions by normalized template after replacing entities, dates, quantities, and metrics with typed slots. No document contributes more than 20 records from one template cluster to one training epoch, and no template cluster can be split across train/dev/test when it originates from a shared generator prompt. Report performance on head entities versus unseen entities and common versus rare templates.

### 15.6 Evaluation quarantine

The following cannot influence source selection, prompts, thresholds, generation, negative mining, or error-driven augmentation: MMLongBench-Doc V1/V2 (including correction logs), SciEGQA-Bench, SlideVQA, LongDocURL, M3DocVQA, MMDocIR expert test, MMDocRAG, DocScope, ChartQAPro, M-LongDoc benchmark, and any official test split designated sealed in §8. Store their hashes in a one-way contamination ledger; developers see only overlap alerts, not questions/content.


## 16. What the models learn at each stage

**Friendly guidepost.** Training is a curriculum, not a data blender. The stages move from valid outputs and simple grounding to evidence sets, cross-page reasoning, synthesis, and finally difficult abstention cases.

### 16.1 Stage definitions

- **Stage 0 — action grammar and grounding initialization:** valid boxes/IDs, one-region selection, empty set, and 20–25% easy two-region examples among answerable records.
- **Stage 1 — same-page evidence sets:** two-to-four required units, table/header, chart/legend, comparison, condition/exception, repeated-value distractors.
- **Stage 2 — cross-page deterministic reasoning:** real same-document arithmetic/comparison/aggregation/temporal/table-text reasoning. Synthetic bundles are tagged and sampled separately.
- **Stage 3 — semi-extractive synthesis:** attributed spans plus connective text; selector and answerer losses remain separate.
- **Stage 4 — fully abstractive multi-evidence:** only evidence-complete explanation/limitation/cause/implication questions.
- **Stage 5 — adversarial/missing-premise/unanswerable:** hard negatives and abstention with ordinary-positive replay.
- **Later RL:** outcome-only training after evidence action grammar is learned. Level-D data never becomes selector gold through reward alone.

### 16.2 Minimum curriculum statistics

| Stage | Unique | Original / synthetic | Zero-evidence share | Cardinality | Answer forms | Supervision | Replay |
| --- | ---: | --- | ---: | --- | --- | --- | --- |
| 0 | 10,000 | 8,000 / 2,000 | 10.0% | 0:1,000; 1:7,000; 2:1,500; 3:500 | Extractive 8,000; numeric 500; list 500; unanswerable 1,000 | A 2,000; B 3,000; C 5,000 | 2,000 replay into Stage 1 |
| 1 | 8,000 | 4,000 / 4,000 | 6.25% | 0:500; 1:1,000; 2:3,500; 3:2,500; 4+:500 | Extractive 3,000; numeric 2,000; list 1,500; yes/no 1,000; unanswerable 500 | A 1,500; B 2,500; C 4,000 | 2,000 Stage-0 replay |
| 2 | 8,000 | 4,000 / 4,000 | 6.25% | 0:500; 1:500; 2:2,500; 3:2,000; 4+:2,500 | Extractive 2,000; numeric 3,500; list 1,500; yes/no 500; unanswerable 500 | A 1,000; B 2,000; C 5,000 | 2,000 prior-stage replay |
| 5 | 6,000 | 2,000 / 4,000 | 66.67% | 0:4,000; 1:1,500; 2:500 | Extractive 1,000; list 500; yes/no 500; unanswerable 4,000 | A 500; B 500; C 5,000 | 2,000 ordinary-positive replay |

### 16.3 Full curriculum statistics

| Stage | Unique | Original / synthetic | Zero-evidence share | Cardinality | Answer forms | Supervision | Replay |
| --- | ---: | --- | ---: | --- | --- | --- | --- |
| 0 | 32,000 | 24,000 / 8,000 | 6.25% | 0:2,000; 1:24,000; 2:6,000 | Extractive 22,000; numeric 4,000; list 2,000; yes/no 2,000; unanswerable 2,000 | A 8,000; B 4,000; C 20,000 | 4,000 replay into Stage 1 |
| 1 | 32,000 | 16,000 / 16,000 | 3.13% | 0:1,000; 1:3,000; 2:16,000; 3:8,000; 4+:4,000 | Extractive 10,000; numeric 8,000; list 6,000; yes/no 2,000; semi 2,000; abstractive 3,000; unanswerable 1,000 | A 6,000; B 8,000; C 18,000 | 4,000 Stage-0 replay |
| 2 | 28,000 | 14,000 / 14,000 | 3.57% | 0:1,000; 1:1,000; 2:10,000; 3:8,000; 4+:8,000 | Extractive 4,000; numeric 10,000; list 4,000; yes/no 1,000; semi 3,000; abstractive 5,000; unanswerable 1,000 | A 4,000; B 10,000; C 14,000 | 4,000 prior-stage replay |
| 3 | 12,000 | 4,000 / 8,000 | 4.17% | 0:500; 1:500; 2:4,000; 3:3,000; 4+:4,000 | Extractive 1,000; numeric 1,000; list 1,000; yes/no 500; semi 7,000; abstractive 1,000; unanswerable 500 | A 0; B 4,000; C 8,000 | 2,000 prior-stage replay |
| 4 | 8,000 | 2,000 / 6,000 | 6.25% | 0:500; 1:500; 2:2,000; 3:2,000; 4+:3,000 | Extractive 500; numeric 500; list 500; yes/no 500; semi 1,000; abstractive 4,500; unanswerable 500 | A 0; B 2,000; C 6,000 | 2,000 prior-stage replay |
| 5 | 16,000 | 4,000 / 12,000 | 75.0% | 0:12,000; 1:2,000; 2:1,000; 3:500; 4+:500 | Extractive 2,000; numeric 500; list 500; yes/no 500; semi 500; unanswerable 12,000 | A 2,000; B 2,000; C 12,000 | 8,000 ordinary-positive replay |

### 16.4 Sampling and optimization fairness

All arms see the same record order, spatial views, image resolution distribution, candidate revision, and source-balanced sampler. Loss scaling may differ only because the output spaces differ; report token/decision counts and effective positive/negative weights. Arms B/Bq/C receive identical candidate tensors and masks. Arm A receives the same underlying pages and selection budget; its boxes are evaluated after canonical normalization rather than snapped to the candidate universe.


## 17. The baselines every new model must beat fairly

**Friendly guidepost.** A new architecture earns its claim only against fair reference systems. The baseline ladder ranges from OCR-text retrieval to multimodal reranking and controlled autoregressive versus parallel selection.

| Baseline | Model class | Input level | Version/license control | Purpose |
| --- | --- | --- | --- | --- |
| BM25 over OCR | Sparse retrieval | Page and candidate text | Frozen OCR normalization; k1/b tuned only on internal dev | Candidate discovery and lexical floor |
| Qwen3-Embedding-0.6B | Dense text embedding | OCR text/semantic unit text | `Qwen/Qwen3-Embedding-0.6B`, Apache-2.0; pin exact HF revision | Text dense retriever |
| Jina v5 omni small retrieval | Multimodal embedding | Page/region image plus text | `jinaai/jina-embeddings-v5-omni-small-retrieval`, CC BY-NC 4.0 | Optional research-only multimodal embedding baseline |
| ColQwen2 v1.0 | Visual late-interaction embedding | Rendered page/region | `vidore/colqwen2-v1.0`, Apache-2.0 | Primary multimodal retriever/miner |
| Qwen3-VL-Reranker-2B pretrained | Multimodal reranker | Question + page/region | Apache-2.0; exact revision pinned | Pretrained reranking floor and Arm-C backbone |
| jina-reranker-m0 | Multimodal reranker | Question + image/text | Qwen2-VL-derived, CC BY-NC 4.0 | Optional noncommercial reranker comparison |
| Task-trained independent scorer | Independent candidate binary scorer | One candidate at a time | Same training records/candidates; no set context | Controls whether gains require set awareness |
| DeepSeek-OCR2-ID | Autoregressive ID selector | Same candidate universe | Arm B | Deployment architecture-plus-backbone comparison |
| Qwen-AR-ID diagnostic | Autoregressive ID selector | Same Qwen backbone/candidates as C | Mandatory Arm Bq | Clean decoder/action comparison |
| MiniVGent-Set | Parallel set selector | All candidates jointly | Arm C | Primary set-aware system |

### 17.1 Model-source pins

- **DeepSeek-OCR-2:** `deepseek-ai/DeepSeek-OCR-2`, Apache-2.0, pinned project revision `aaa02f3811945a91062062994c5c4a3f4c0af2b0`. [DSOCR2-HF] [DSOCR2-P]
- **Qwen3-VL-Reranker-2B:** `Qwen/Qwen3-VL-Reranker-2B`, Apache-2.0, approximately 2.13B parameters; resolve and record the full immutable commit at acquisition. [QWEN-RERANK-HF] [QWEN-RERANK-P]
- **Qwen3-Embedding-0.6B:** dense text embedding, not a reranker. [QWEN-EMB-HF]
- **jina-embeddings-v5-omni-small-retrieval:** multimodal embedding model, not a cross-encoder; research use is constrained by CC BY-NC 4.0. [JINA-EMB-HF]
- **jina-reranker-m0:** multimodal reranker/cross-encoder-like scoring model, not an embedding index; CC BY-NC 4.0. [JINA-M0-HF]
- **ColQwen2 v1.0:** late-interaction visual retriever, not a generative reranker. [COLQWEN-HF]

### 17.2 Mining versus truth

Retrievers and rerankers may propose hard candidates. They may not label them. All final evidence judgments use the original, unmodified question and the independent verification pipeline. Report **candidate recall before reranking**, **strict complete-evidence oracle recall**, and **final evidence precision** separately; a miner can improve the first while harming the latter.


## 18. How we decide whether a result is trustworthy

**Friendly guidepost.** Accuracy alone is not enough. The gates here test whether the candidate generator could have found the answer, whether the complete evidence set was recovered, and whether downstream answers really depend on that evidence.

### 18.1 Metrics

| Metric | Definition |
| --- | --- |
| Candidate atomic oracle recall | Fraction of gold atomic evidence units overlapped/mapped by at least one candidate; max over valid alternative groups. |
| Strict all-required-evidence oracle recall | Fraction of questions for which one valid complete evidence group is fully representable in the candidate universe. |
| Evidence precision/recall/F1 | Candidate-ID exact set metrics for B/Bq/C; box matching for A uses Hungarian assignment at IoU≥0.5 plus text/role equivalence. |
| Strict complete-evidence recall | Question-level indicator that all required evidence and no prohibited/contradictory evidence is selected; report with and without precision constraint. |
| Cardinality accuracy | Exact predicted evidence count; also mean absolute cardinality error. |
| No-evidence false-positive rate | Fraction of zero-evidence records with any selected region/ID. |
| Selected-evidence answer accuracy | Fixed downstream answerer sees only selected evidence; score exact/F1/program accuracy or source-appropriate metric. |
| Leave-one-evidence-out degradation | Answer-score drop when each selected gold evidence item is removed; validates necessity and downstream use. |
| Selected area/token cost | Union area ratio, number of pages/candidates, rendered visual tokens, OCR tokens, and downstream latency/cost. |
| Box validity rates | Malformed, repaired, duplicate, out-of-bounds, oversized (>50% page without gold justification), and zero-area box rates. |
| Hard-negative accuracy | Correct rejection of verified hard negatives, by family. |
| False-negative audit rate | Human-confirmed relevant examples among training negatives, with Wilson interval. |
| Stratified performance | Real versus synthetic; extractive versus semi-/abstractive; single-page versus real cross-page versus synthetic bundle; source, answer type, reasoning, cardinality, supervision strength. |

### 18.2 Project advancement gates

These are **Our proposed plan** (`proposed_design`) thresholds, not published community standards.

| Gate | Threshold |
| --- | --- |
| Candidate universe precondition | Atomic oracle recall ≥98.5% overall; strict complete oracle ≥95% overall, ≥92% real cross-page, and ≥90% chart/header strata. p95 candidates ≤512/page and ≤2,048/bundle. |
| Stage 0 advance | Evidence F1 ≥0.85; empty-set specificity ≥0.95; malformed output <0.25%. |
| Stage 1 advance | Strict complete-evidence recall ≥0.70; cardinality accuracy ≥0.75 on same-page multi-evidence. |
| Stage 2 advance | Real cross-page strict recall ≥0.60; selected-only answer accuracy ≥70% of oracle-evidence answerer; leave-one-out degradation ≥10 percentage points. |
| Stage 3/4 advance | Every answer factual claim supported; unsupported-claim rate ≤2% in human audit; evidence precision not reduced by >3 points versus Stage-2 checkpoint. |
| Stage 5 advance | Hard-negative accuracy ≥0.80; no-evidence FPR <5%; false-negative upper 95% bound ≤2%. |
| Architecture interpretation | If candidate strict oracle is below gate in a stratum, MiniVGent errors there are reported as candidate-limited and excluded from claims about selector capacity. |
| Statistical reporting | Three training seeds; paired document bootstrap with 10,000 replicates; 95% CIs; Holm correction for the pre-registered primary contrasts A–B, Bq–C, and A–C. |

### 18.3 Fixed answerer protocol

Use one frozen downstream answerer, one frozen answer prompt, and identical evidence serialization for all arms. Evidence localization is scored before answer generation. The answerer also runs under three controlled contexts: gold evidence, selected evidence, and full candidate bundle. This separates selector failure from answerer failure and quantifies the value/cost of localization.

### 18.4 Box-versus-ID comparability

For Arm A, convert predicted boxes to pixel-normalized geometry, merge duplicates at IoU≥0.9, and match to gold roles. For cost comparison, also compute the candidates overlapped by each predicted box **after** prediction; this diagnostic does not change Arm-A training or primary box metrics. Selection budgets are enforced as maximum union area/tokens, not equal numbers of actions, because raw boxes and IDs have different granularity.


## 19. Controlled ablations — changing one thing at a time

**Friendly guidepost.** Ablations are the scientific version of changing one ingredient at a time. Each comparison has one hypothesis and one control so that a gain has a plausible explanation.

| Ablation | Pre-registered hypothesis | Valid control |
| --- | --- | --- |
| No synthetic augmentation | Verified synthetic multi-evidence improves strict complete recall beyond more native single-hop data. | Same original 64k; replace synthetic records with hash-ranked native/replayed originals; same presentations. |
| Positive paraphrases only | Lexical robustness improves without evidence-set gains. | Native originals + paraphrase quota; match total with repeated originals, not other synthetic types. |
| Multi-evidence augmentation only | Necessity-verified multi-evidence drives set behavior. | Native originals + same-page/real-cross-page; no paraphrase/negative generation. |
| Mined negatives only | Conventional near misses suffice without query perturbation. | Same negative count/source/rank strata; no generated query negatives. |
| Generated query-side negatives only | Semantic perturbations improve fine-grained rejection but may overfit artifacts. | Same negative count; remove mined negatives; ordinary positives unchanged. |
| Combined negative curriculum | Mined and generated negatives are complementary. | Union at fixed total, 50/50; compare to each single-family control. |
| Native evidence only | Level-C pseudo-label noise may erase scale benefits. | Use only Level A/B records; downsample all arms identically. |
| Native + verified pseudo-evidence | Verifier-gated Level C adds useful diversity. | Same native core plus accepted C; compare per source and supervision stratum. |
| Extractive-only SFT | Abstractive training may hurt localization precision. | Remove Stages 3/4 and answerer-synthesis objectives; match presentations with extractive replay. |
| Mixed extractive/abstractive curriculum | Later synthesis improves realistic questions without catastrophic forgetting. | Full staged schedule versus extractive-only at matched update count. |
| No single-hop replay | Replay prevents forgetting of basic grounding/abstention. | Remove replay but preserve stage-unique data; report Stage-0 regression. |
| Real same-document cross-page versus synthetic bundles | Synthetic bundles do not fully transfer to document coherence. | Train matched 4k real-only versus 4k synthetic-only and test separately on real/synthetic. |
| DeepSeek-OCR2-ID versus MiniVGent-Set | Deployment systems differ in more than decoder because backbones differ. | Same candidates/data/budgets; interpret only as architecture-plus-backbone. |
| Qwen-AR-ID versus MiniVGent-Set | Parallel set prediction improves complete-set recall/cardinality under the same backbone. | Same Qwen reranker initialization, inputs, candidates, and updates; only output head/decoder differs. |
| DeepSeek-OCR2-Box versus DeepSeek-OCR2-ID | Raw coordinates trade flexible recall for malformed/oversized output and action complexity. | Same DeepSeek initialization/pages/questions; ID arm receives frozen candidates. |
| Answer-only RL with versus without prior multi-evidence SFT | Outcome RL is safer/more sample-efficient after evidence action grammar. | Same RL data/reward/budget; initialize one from full SFT and one from Stage-0/single-hop checkpoint. |

Primary ablations use the same document groups, seeds, number of optimizer updates, maximum visual tokens, candidate revision, answerer, and evaluation sets. When a data family is removed, replacement examples are replayed from the **same source/topology where possible**; the paper must distinguish a fixed-update comparison from a fixed-unique-data comparison.


## 20. The practical order for acquiring and preparing the data

**Friendly guidepost.** This is the build order. It starts with rights, revisions, and manifests; moves through deduplication and candidate-oracle checks; and delays expensive augmentation until the foundations are trustworthy.

| Step | Action | Required artifact |
| --- | --- | --- |
| 0. Governance freeze | Approve this nonbinding specification; confirm it does not modify the active Visual-CoT foundation contract. Assign data owner, legal owner, evaluation custodian, and human-audit lead. | Signed decision log |
| 1. Source ledger acquisition | Record URL, paper version, repository/HF commit, license text, file names, sizes, and cryptographic hashes. No model training. | Immutable `source_ledger.json` |
| 2. Evaluation quarantine | Custodian hashes sealed assets and exposes only duplicate alerts. Developers never inspect questions. | Quarantine manifest |
| 3. Metadata-only identity graph | Build document/page/question provenance graph and cross-dataset duplicate clusters. | `identity_groups.parquet` + audit report |
| 4. License/rights gate | Approve or remove Visual-CoT views, BoundingDocs parents, RefChartQA, QASPER/PeerQA PDFs, and conditional RL sources. | Per-source use decision |
| 5. Deterministic rendering/OCR | Pin DeepSeek-OCR-2 revision/prompt/rendering; produce page hashes and OCR/layout metadata. | Reproducible normalized pages |
| 6. Frozen semantic candidates | Run one parser revision over every allowed page; add only approved fallbacks; measure oracle coverage before model training. | `candidate_revision` + oracle report |
| 7. Native/aligned manifest | Map Level A/B labels and audit Level C source labels; select original quotas by hash sampler. | Original train/dev/test manifests |
| 8. Pilot generation | Generate 10% of each augmentation family; run all verifiers/human audits; revise only on train pilot. | Pilot yield/quality report |
| 9. Full generation and backfill | Generate until exact accepted quotas or declare infeasible; no forced acceptance. | Derivative manifests + reject logs |
| 10. Manifest freeze | Validate arithmetic, schema, provenance DAG, split purity, license tags, and no sealed overlap. | Signed corpus release candidate |
| 11. Baseline-only dry run | Run oracle/BM25/embedding/reranker metrics to verify evaluability; still no architecture claims. | Dataset readiness report |
| 12. Future training authorization | A separate owner decision may authorize SFT/RL. This document itself does not. | New implementation/training contract |

The active Visual-CoT project remains limited to its already authorized data-foundation stages. Steps 5–12 for this proposed three-arm experiment must not be inferred as authorized by work on that separate project.


## 21. Risks, open questions, and choices that still need owner approval

**Friendly guidepost.** These are not footnotes to wave away. They are the exact uncertainties, licensing questions, release gaps, and architecture choices that could change what the experiment is allowed to claim.

### 21.1 Risk register

| Risk | Failure mode | Control |
| --- | --- | --- |
| License/redistribution | Visual-CoT conflict; SlideVQA restriction; RefChartQA copyleft; missing SciEGQA/M-LongDoc terms | Legal owner can remove sources without altering evaluation; invoke fixed backfill |
| Backbone confound | DeepSeek Arm B versus Qwen Arm C cannot isolate decoder | Mandatory Qwen-AR-ID diagnostic; qualify claims |
| Candidate ceiling | MiniVGent cannot select evidence absent from frozen candidates | Oracle gates before training; candidate-limited strata separated |
| Incomplete native evidence | Answer boxes/pages omit headers, qualifiers, or alternative support | Level-C downgrade, ignored candidates, subset verification, human audit |
| Synthetic shortcuts | Generator wording, source patterns, or bundle metadata reveal labels | Naturalness/shortcut judges, prompt-family split, real-vs-synthetic reporting |
| False negatives | Unselected context may still be useful or answer the altered query elsewhere | Document-wide checks, `ignore`, human false-negative gate |
| Cross-dataset leakage | Derivative/mixture datasets share questions, pages, reports, and Wikipedia sources | Identity graph and canonical source ownership |
| Benchmark contamination | Public long-document benchmarks may appear in training/pretraining | Sealed custodian, document hashes, no error-driven generation |
| Evidence necessity versus model prior | A model may answer with world knowledge despite missing evidence | Entailment plus answerer tests; ambiguous records ignored |
| Cost imbalance | Arm A can output arbitrary area while ID/Set select atomic units | Area/token budgets and dual cost metrics |
| Question realism | Fluent generated questions may remain contrived | Dependency taxonomy, human naturalness audit, reader/expert-authored source strata |
| Curriculum forgetting | Abstractive/adversarial stages may erode basic grounding | Fixed replay and stagewise regression gates |

### 21.2 Required owner approvals

| ID | Decision |
| --- | --- |
| A1 | Authorize exact source license interpretations, especially Visual-CoT, BoundingDocs parents, RefChartQA, QASPER/PeerQA PDFs, SciEGQA, and M-LongDoc. |
| A2 | Accept the mandatory Qwen-AR-ID diagnostic as part of any decoder/action-space claim. |
| A3 | Approve the 32k minimum or 128k full selector plan; changing totals requires a versioned amendment. |
| A4 | Approve DeepSeek-OCR-2 revision/prompt and the frozen semantic-candidate parser/fallback policy. |
| A5 | Approve verifier models/prompts and human-review budget before generation. |
| A6 | Approve whether RefChartQA remains; if denied, accept the documented chart-grounding gap. |
| A7 | Approve any SciEGQA train addendum only after explicit rights and audit. |
| A8 | Approve conditional later RL data and rewards in a separate contract. |
| A9 | Approve project advancement gates or replace them before test access; never tune them on sealed evaluation. |

### 21.3 Claims that remain hypotheses

- DeepSeek-OCR2 coordinate generation has a practical advantage when the candidate oracle ceiling is low.
- Autoregressive ID selection under-selects distributed evidence relative to parallel set prediction.
- Qwen reranker initialization transfers relevance information into a proposal/set decoder.
- Verified synthetic multi-evidence improves real same-document cross-page performance.
- Semi-/fully-abstractive curricula improve evidence selection rather than only answer fluency.
- Generated query negatives improve fine semantic discrimination without increasing false-negative supervision.

No cited dataset establishes these architecture claims. They are tested only by the controlled arms and ablations above.


## 22. Source ledger and bibliography — the research paper trail

**Friendly guidepost.** Every important factual claim should lead back here. The ledger records the primary papers, repositories, cards, licenses, releases, versions, and access dates used by the specification.

All web/repository sources in this report were accessed **2026-08-06**. The version/revision column is the required acquisition pin where known; entries marked “accessed” must be resolved to a full immutable commit or dataset revision before manifest construction.

| Citation key | Primary source | Version/revision observed | Official URL |
| --- | --- | --- | --- |
| VCOT-P | Visual-CoT paper | arXiv v3 | https://arxiv.org/pdf/2403.16999 |
| VCOT-GH | Visual-CoT repository | commit 83212ae474ab46b70048a6df6a386cff6589ff92 | https://github.com/deepcs233/Visual-CoT |
| VCOT-HF | Visual-CoT HF | revision 223d2d8c1146fda2bb918801b8276c587b78b61c observed | https://huggingface.co/datasets/deepcs233/Visual-CoT |
| DOCVQA-P | DocVQA paper | 2020 | https://arxiv.org/pdf/2007.00398 |
| DOCVQA-SITE | DocVQA portal | accessed 2026-08-06 | https://rrc.cvc.uab.es/?ch=17 |
| INFO-P | InfographicVQA paper | 2021 | https://arxiv.org/pdf/2104.12756 |
| INFO-SITE | InfographicVQA portal | accessed 2026-08-06 | https://rrc.cvc.uab.es/?ch=17 |
| SROIE-P | SROIE paper | ICDAR 2019 | https://arxiv.org/pdf/2103.10213 |
| SROIE-SITE | SROIE portal | accessed 2026-08-06 | https://rrc.cvc.uab.es/?ch=13 |
| TEXTVQA-P | TextVQA paper | 2019 | https://arxiv.org/pdf/1904.08920 |
| TEXTVQA-SITE | TextVQA site | accessed 2026-08-06 | https://textvqa.org/ |
| TEXTCAPS-P | TextCaps paper | 2020 | https://arxiv.org/pdf/2003.12462 |
| TEXTCAPS-SITE | TextCaps site | accessed 2026-08-06 | https://textvqa.org/textcaps/ |
| BOUND-P | BoundingDocs paper | 2025 | https://arxiv.org/pdf/2501.03403 |
| BOUND-HF | BoundingDocs HF | v2.0 | https://huggingface.co/datasets/letxbe/BoundingDocs |
| BBOX-V1 | BBox DocVQA historic paper | arXiv v1 | https://arxiv.org/pdf/2511.15090v1 |
| SCIEGQA-P | SciEGQA paper | arXiv v2, 2026-03-30 | https://arxiv.org/pdf/2511.15090 |
| SCIEGQA-PROJ | SciEGQA project | accessed 2026-08-06 | https://yuwenhan07.github.io/SciEGQA-project/ |
| SCIEGQA-GH | SciEGQA repo | accessed 2026-08-06 | https://github.com/yuwenhan07/SciEGQA |
| SCIEGQA-TRAIN | SciEGQA-Train HF | 30,780 rows observed | https://huggingface.co/datasets/Yuwh07/SciEGQA-Train |
| SCIEGQA-BENCH | SciEGQA-Bench HF | 1,623 rows observed | https://huggingface.co/datasets/Yuwh07/SciEGQA-Bench |
| M3G-PROJ | M3Grounder project | Dataset Soon at access | https://m3grounder.github.io/ |
| M3G-P | M3Grounder paper | CVPR 2026 | https://openaccess.thecvf.com/content/CVPR2026/html/Venna_M3Grounder_Mask-Based_Multi-Span_and_Multi-Granular_Grounding_for_Document_QA_CVPR_2026_paper.html |
| MPDOC-P | MP-DocVQA paper | 2022 | https://arxiv.org/pdf/2212.05935 |
| MPDOC-GH | MP-DocVQA framework | accessed 2026-08-06 | https://github.com/rubenpt91/MP-DocVQA-Framework |
| MPDOC-HF | MP-DocVQA HF mirror/card | accessed 2026-08-06 | https://huggingface.co/datasets/naver-clova-ix/MP-DocVQA |
| DUDE-P | DUDE paper | 2023 | https://arxiv.org/pdf/2305.08455 |
| DUDE-GH | DUDE repo | main SHA 540a746… README snapshot | https://github.com/duchallenge-team/dude |
| DUDE-HF | DUDE loader/card | accessed 2026-08-06 | https://huggingface.co/datasets/jordyvl/DUDE_loader |
| SLIDE-P | SlideVQA paper | 2023 | https://arxiv.org/pdf/2301.04883 |
| SLIDE-GH | SlideVQA repo | main accessed 2026-08-06 | https://github.com/nttmdlab-nlp/SlideVQA |
| SLIDE-LIC | SlideVQA license | evaluation/internal only | https://github.com/nttmdlab-nlp/SlideVQA/blob/main/LICENSE |
| QASPER-P | QASPER paper | NAACL 2021 | https://aclanthology.org/2021.naacl-main.365.pdf |
| QASPER-HF | QASPER HF | updated 2022-10-07 | https://huggingface.co/datasets/allenai/qasper |
| FINQA-P | FinQA paper | EMNLP 2021 | https://aclanthology.org/2021.emnlp-main.300.pdf |
| FINQA-GH | FinQA repo | train SHA a6c2c5a… | https://github.com/czyssrs/FinQA |
| TATQA-P | TAT-QA paper | ACL 2021 | https://aclanthology.org/2021.acl-long.254.pdf |
| TATQA-GH | TAT-QA repo | commit 870accc41953dcde885aabeb963d94aabdc0fbc3 observed | https://github.com/NExTplusplus/TAT-QA |
| MULTIHIERTT-P | MultiHiertt paper | 2022 | https://arxiv.org/pdf/2206.01347 |
| MULTIHIERTT-GH | MultiHiertt repo | commit 45bd9ccdf3142ea059bd5e69c0afb83437fa539c observed | https://github.com/psunlpgroup/MultiHiertt |
| COND-P | ConditionalQA paper | 2021 | https://arxiv.org/pdf/2110.06884 |
| COND-GH | ConditionalQA repo | v1.0 | https://github.com/haitian-sun/ConditionalQA |
| MESAQA-P | MESAQA paper | COLING 2025 | https://aclanthology.org/2025.coling-main.724.pdf |
| MESAQA-GH | MESAQA repo | README SHA 068f0c4… | https://github.com/reiiwang/MESAQA |
| MESAQA-HF | MESAQA HF | updated 2025-06-17 | https://huggingface.co/datasets/riiwang/MESAQA |
| QUOTESUM-P | SEMQA paper | NAACL 2024 | https://arxiv.org/pdf/2311.04886 |
| QUOTESUM-GH | QuoteSum repo | v1; archived | https://github.com/google-research-datasets/QuoteSum |
| MLONG-P | M-LongDoc paper | 2024 | https://arxiv.org/pdf/2411.06176 |
| MLONG-GH | M-LongDoc repo | latest observed 944a61950… | https://github.com/kenchan0226/multimodal-docs-public |
| MMQA-P | MultiModalQA paper | 2021 | https://arxiv.org/pdf/2104.06039 |
| MMQA-GH | MultiModalQA repo | accessed 2026-08-06 | https://github.com/allenai/multimodalqa |
| HOTPOT-P | HotpotQA paper | 2018 | https://arxiv.org/pdf/1809.09600 |
| HOTPOT-SITE | HotpotQA site | accessed 2026-08-06 | https://hotpotqa.github.io/ |
| BREAK-P | BREAK paper | 2020 | https://arxiv.org/pdf/2001.11770 |
| BREAK-GH | BREAK official dataset page | accessed 2026-08-06 | https://allenai.org/data/break |
| TATDQA-P | TAT-DQA paper | arXiv v3 | https://arxiv.org/pdf/2207.11871 |
| TATDQA-GH | TAT-DQA repo | master accessed 2026-08-06 | https://github.com/NExTplusplus/TAT-DQA |
| TATDQA-HF | TAT-DQA HF | accessed 2026-08-06 | https://huggingface.co/datasets/next-tat/TAT-DQA |
| PEERQA-P | PeerQA paper | NAACL 2025 | https://aclanthology.org/2025.naacl-long.22.pdf |
| PEERQA-GH | PeerQA repo | v1.0, latest observed 2025-06-23 | https://github.com/UKPLab/PeerQA |
| REFCHART-P | RefChartQA paper | 2025 | https://arxiv.org/pdf/2503.23131 |
| REFCHART-GH | RefChartQA repo | main accessed 2026-08-06 | https://github.com/moured/RefChartQA |
| REFCHART-HF | RefChartQA HF | 73,702 current rows | https://huggingface.co/datasets/omoured/RefChartQA |
| SPIQA-P | SPIQA paper | NeurIPS 2024 | https://arxiv.org/pdf/2407.09413 |
| SPIQA-HF | SPIQA HF | accessed 2026-08-06 | https://huggingface.co/datasets/google/spiqa |
| HITAB-P | HiTab paper | ACL 2022 | https://aclanthology.org/2022.acl-long.78.pdf |
| HITAB-GH | HiTab repo | main accessed 2026-08-06 | https://github.com/microsoft/HiTab |
| CHARTQA-P | ChartQA paper | ACL Findings 2022 | https://aclanthology.org/2022.findings-acl.177.pdf |
| CHARTQA-GH | ChartQA repo | main accessed 2026-08-06 | https://github.com/vis-nlp/ChartQA |
| LONGDOCURL-P | LongDocURL paper | ACL 2025 | https://aclanthology.org/2025.acl-long.57.pdf |
| LONGDOCURL-PROJ | LongDocURL project | accessed 2026-08-06 | https://longdocurl.github.io/ |
| M3DOC-P | M3DocRAG paper | accessed 2026-08-06 | https://arxiv.org/pdf/2411.04952 |
| M3DOC-PROJ | M3DocRAG project | accessed 2026-08-06 | https://m3docrag.github.io/ |
| MMDOCIR-P | MMDocIR paper | 2025 | https://arxiv.org/pdf/2501.08828 |
| MMDOCIR-GH | MMDocIR repo | accessed 2026-08-06 | https://github.com/texttron/MMDocIR |
| MMDORAG-P | MMDocRAG paper | 2025 | https://arxiv.org/pdf/2505.16470 |
| MMDORAG-GH | MMDocRAG repo | accessed 2026-08-06 | https://github.com/Kudux/MMDocRAG |
| MMLBD-P | MMLongBench-Doc V1 paper | 2024 | https://arxiv.org/pdf/2404.15754 |
| MMLBD-GH | MMLongBench-Doc V1 repo | accessed 2026-08-06 | https://github.com/yubo-ma/MMLongBench-Doc |
| MMLBD2-P | MMLongBench-Doc V2 paper | released 2026-08-04 | https://arxiv.org/abs/2608.03397 |
| MMLBD2-GH | MMLongBench-Doc V2 repo | accessed 2026-08-06 | https://github.com/meiu-zhang/MMlongBench-Doc |
| MMLBD2-HF | MMLongBench-Doc V2 HF | accessed 2026-08-06 | https://huggingface.co/datasets/meiu/MMlongBench-Doc |
| DOCSCOPE-P | DocScope paper | 2026 | https://arxiv.org/abs/2605.08888 |
| DOCSCOPE-PROJ | DocScope project | unreleased at access | https://mililab.github.io/DocScope/ |
| CHARTQAPRO-P | ChartQAPro paper/project paper link | accessed 2026-08-06 | https://vis-nlp.github.io/ChartQAPro/ |
| CHARTQAPRO-PROJ | ChartQAPro project | accessed 2026-08-06 | https://vis-nlp.github.io/ChartQAPro/ |
| DOCFINQA-P | DocFinQA project | accessed 2026-08-06 | https://gaspar.ai/resources/docfinqa/ |
| DOCFINQA-GH | DocFinQA repo | accessed 2026-08-06 | https://github.com/ibm/DocFinQA |
| DOCTRACE-P | DocTrace paper | released 2026-08-04 | https://arxiv.org/pdf/2608.03292 |
| DOCRE-P | DocReRank paper | 2025 | https://arxiv.org/pdf/2505.22584 |
| DOCRE-PROJ | DocReRank project | accessed 2026-08-06 | https://navvewas.github.io/DocReRank/ |
| DSOCR2-HF | DeepSeek-OCR-2 model card | revision aaa02f3811945a91062062994c5c4a3f4c0af2b0 | https://huggingface.co/deepseek-ai/DeepSeek-OCR-2 |
| DSOCR2-P | DeepSeek-OCR-2 paper | 2026 | https://arxiv.org/pdf/2601.20552 |
| QWEN-RERANK-HF | Qwen3-VL-Reranker-2B model card | accessed 2026-08-06 | https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B |
| QWEN-RERANK-P | Qwen3-VL embedding/reranker paper | 2026 | https://arxiv.org/pdf/2601.04720 |
| QWEN-EMB-HF | Qwen3-Embedding-0.6B model card | accessed 2026-08-06 | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B |
| JINA-EMB-HF | Jina v5 omni retrieval model card | accessed 2026-08-06 | https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-retrieval |
| JINA-M0-HF | jina-reranker-m0 model card | accessed 2026-08-06 | https://huggingface.co/jinaai/jina-reranker-m0 |
| COLQWEN-HF | ColQwen2 v1.0 model card | accessed 2026-08-06 | https://huggingface.co/vidore/colqwen2-v1.0 |


## Appendix A. Every dataset at a glance

**Appendix guide.** Use this table for the one-screen view of every audited source and its experimental role.

| Dataset | Domain/type | Published scale | Evidence supplied | Strength | Decision |
| --- | --- | --- | --- | --- | --- |
| Visual-CoT | Derivative spatial QA | Paper-rounded 26k/20k/15k/4k/18k/32k; pinned rows **26,117/19,429/6,088/2,584/17,280/33,866** | Answer boxes | C | Core/aux by source |
| DocVQA | Single-page documents | 50,000 Q / 12,767 images | OCR, Q/A | D→C via Visual-CoT | Exclude direct |
| InfographicVQA | Infographics | 30,035 / 5,485 | OCR, Q/A | D→C via Visual-CoT | Exclude direct |
| SROIE | Receipts | 1,000 images; 0 native Q | Line boxes/KIE | A for fields | Exclude direct |
| TextVQA | Scene text | 45,336 / 28,408 | OCR, Q/A | D | Exclude |
| BoundingDocs | Unified docs | 249,016 Q / 48,151 docs / 237,437 pages | Answer boxes | C | Auxiliary |
| SciEGQA | Scientific pages | 30,780 train; 1,623 bench | Boxes/pages | A/C | Bench sealed; train excluded |
| GroundingDocQA | Mask grounding | ~2M reported | Masks | A claimed | Unavailable |
| MP-DocVQA | Multi-page docs | **46,176 questions; 6,000 documents** | Gold answer page | B page | Development |
| DUDE | Diverse multi-page | 41,541 abstract / 41,491 table; 5,019/4,974 docs | Boxes for extractive | A/B/C | Core |
| SlideVQA | Slide decks | 14,484 Q / 2,619 decks / 52,586 slides | Evidence pages/nodes | A/B | Sealed |
| QASPER | Scientific papers | 5,049 Q / 1,585 papers | Evidence paragraphs | B | Auxiliary aligned |
| FinQA | Financial reasoning | 8,281 Q / 2,789 reports | Facts/programs | B | Planner |
| TAT-QA | Table+text finance | 16,552 Q / 2,757 contexts | Facts/programs | B | Planner |
| TAT-DQA | Visual financial QA | 16,558 Q / 2,758 docs / 3,067 pages | Boxes/evidence/programs | A/B | Core |
| MultiHiertt | Multi-table finance | 10,440 Q / 2,513 reports | Facts/programs | B | Planner |
| ConditionalQA | Policy QA | 3,427 Q | Evidence spans | B | Planner |
| MESAQA | Multi-span text | 6,183 derived | Sentence IDs | B/C | Exclude |
| QuoteSum | Semi-extractive multi-source | 1,376 Q / 4,009 answers | Attributed spans | B | Answerer |
| M-LongDoc | Long multimodal docs | 1,051 benchmark Q / 180 docs | Weak/varied | D/C | Sealed/conditional RL |
| MultiModalQA | Multimodal multi-source | 29,918 Q | Supporting contexts | B | Planner |
| HotpotQA | Text multi-hop | ~113k Q | Supporting facts | B | Planner |
| BREAK | Decomposition | 83,978 Q | QDMR | N/A | Planner |
| PeerQA | Peer-review scientific QA | 579 Q / 208 papers | Passage relevance | B | Auxiliary aligned |
| RefChartQA | Chart grounding | 73,702 Q | Element boxes | A/C | Auxiliary conditional |
| SPIQA | Scientific multimodal QA | ~270k train / 11,820 test | Reference figures/text | B/D | Answerer |
| HiTab | Hierarchical tables | 10,672 release / 10,686 paper | Programs/cells | B | Planner |
| LongDocURL | Long PDFs | 2,325 Q / 396 PDFs / >33k pages | Varied | D/B | Sealed |
| M3DocVQA | Rendered multi-doc | 2,441 Q / 3,368 PDFs / >41k pages | Supporting docs/pages | B | Sealed |
| MMDocIR | Retrieval benchmark | 73,843 train; 313 expert test | Page labels | B | Sealed |
| MMDocRAG | RAG benchmark | 4,055 Q / 2,965 docs / 236,230 pages | 1–3 evidence items | B | Sealed |
| MMLongBench-Doc V2 | Long-doc benchmark | 1,071 Q / 134 docs | Per-page evidence | B | Sealed |
| DocScope | Rationale benchmark | 1,124 Q / 262 docs | Claimed page/region/fact | A/B claimed | Sealed unresolved |
| ChartQAPro | Chart benchmark | 1,948 Q / 1,948 charts | Answer only | D | Sealed |
| DocFinQA | Full financial reports | ~7,437/7,621 Q / 483 docs | Inherited facts/programs | B | Development |


## Appendix B. Where every planned record goes

**Appendix guide.** These tables reconcile the exact minimum and full-plan allocations.

### B.1 Minimum selector corpus

| Source | Original | Accepted synthetic | Final |
| --- | ---: | ---: | ---: |
| Visual-CoT DocVQA | 3,000 | 2,000 | 5,000 |
| Visual-CoT InfographicVQA | 2,000 | 1,800 | 3,800 |
| Visual-CoT SROIE | 500 | 500 | 1,000 |
| DUDE | 4,500 | 4,200 | 8,700 |
| TAT-DQA | 5,000 | 3,700 | 8,700 |
| BoundingDocs source-distinct | 1,000 | 600 | 1,600 |
| QASPER aligned | 1,000 | 700 | 1,700 |
| RefChartQA | 1,000 | 500 | 1,500 |
| **Total** | **18,000** | **14,000** | **32,000** |

### B.2 Full selector corpus

| Source | Original | Accepted synthetic | Final |
| --- | ---: | ---: | ---: |
| Visual-CoT DocVQA | **16,000** | 9,600 | **25,600** |
| Visual-CoT InfographicVQA | 9,000 | 7,500 | 16,500 |
| Visual-CoT SROIE | **2,000** | 2,100 | **4,100** |
| DUDE | 12,500 | 13,400 | 25,900 |
| TAT-DQA | 12,000 | 15,100 | 27,100 |
| BoundingDocs source-distinct | 5,000 | 3,300 | 8,300 |
| QASPER aligned | 1,100 | 6,200 | 7,300 |
| PeerQA aligned | 400 | 2,200 | 2,600 |
| RefChartQA | 6,000 | 4,600 | 10,600 |
| **Total** | **64,000** | **64,000** | **128,000** |

### B.3 Full future program

| Pool | Records |
| --- | ---: |
| Selector SFT | 128,000 |
| Planner/decomposition | 32,000 |
| Answerer SFT | 16,000 |
| Conditional later RL | 20,000 |
| **Total** | **196,000** |


## Appendix C. How the augmentation numbers add up

**Appendix guide.** This is the augmentation ledger: generated candidates, expected or measured yield, accepted targets, and final totals.

| Family | Minimum generated | Minimum accepted | Full generated | Full accepted |
| --- | ---: | ---: | ---: | ---: |
| Paraphrase | 2,500 | 2,000 | 10,000 | 8,000 |
| Same-page multi | 8,000 | 4,000 | 24,000 | 12,000 |
| Question composition | 0 | 0 | 15,000 | 6,000 |
| Real cross-page | 10,000 | 4,000 | 25,000 | 10,000 |
| Synthetic bundle | 0 | 0 | 10,000 | 4,000 |
| Semi-extractive | 0 | 0 | 16,000 | 8,000 |
| Abstractive | 0 | 0 | 15,000 | 6,000 |
| Missing premise | 4,000 | 2,000 | 10,000 | 5,000 |
| Query negative | 8,000 | 2,000 | 20,000 | 5,000 |
| **Total** | **32,500** | **14,000** | **145,000** | **64,000** |


## Appendix D. What each training stage contains

**Appendix guide.** This table shows which mixture reaches each training stage.

| Plan | Stage | Unique records | Replay presentations | Destination |
| --- | --- | ---: | ---: | --- |
| Minimum | 0 | 10,000 | 0 | Action grammar |
| Minimum | 1 | 8,000 | 2,000 | Same-page sets |
| Minimum | 2 | 8,000 | 2,000 | Real cross-page |
| Minimum | 5 | 6,000 | 2,000 | Adversarial/abstention |
| Minimum total |  | 32,000 | 6,000 | 36,000 presentations |
| Full | 0 | 32,000 | 0 | Action grammar |
| Full | 1 | 32,000 | 4,000 | Same-page sets |
| Full | 2 | 28,000 | 4,000 | Cross-page deterministic |
| Full | 3 | 12,000 | 2,000 | Semi-extractive |
| Full | 4 | 8,000 | 2,000 | Abstractive |
| Full | 5 | 16,000 | 8,000 | Adversarial/abstention |
| Full total |  | 128,000 | 20,000 | 148,000 presentations |


## Appendix E. What can be acquired and how it may be used

**Appendix guide.** Check this before downloading or training. Availability does not automatically grant training or redistribution rights.

| Source | Availability | License/terms finding | Protocol status |
| --- | --- | --- | --- |
| Visual-CoT | Public | Conflicting Apache-2.0 vs CC BY-NC/research-only; parent terms | Conditional |
| DocVQA/InfographicVQA/MP-DocVQA/DUDE | RRC registration | RRC terms; DUDE annotations/loader CC BY 4.0 | Conditional by source |
| SROIE | Competition portal | Official competition terms | Conditional |
| TextVQA | Public | CC BY 4.0 annotations + Open Images | Audit-only exclusion; not acquired or used |
| BoundingDocs | HF public | CC BY 4.0 + parents | Conditional source-distinct |
| SciEGQA | HF public | No explicit license located | Train denied; bench sealed |
| GroundingDocQA | Not released | No license | Denied |
| SlideVQA | Public repo | Evaluation/internal only | Evaluation only |
| QASPER | HF public | CC BY 4.0 annotations; paper rights separate | Conditional alignment |
| FinQA/TAT-QA/HiTab | Public | MIT code/release; source reports vary | Planner conditional |
| TAT-DQA | Public | CC BY 4.0 | Allowed |
| MultiHiertt | Public | Apache-2.0 repo; reports vary | Planner conditional |
| ConditionalQA | Public | CC BY-NC-SA 4.0 | Noncommercial planner |
| MESAQA | Public | Unclear | Denied |
| QuoteSum | Public | CC BY-SA 4.0 | Allowed answerer with obligations |
| PeerQA | Public | CC BY-NC-SA 4.0 | Noncommercial auxiliary |
| RefChartQA | Public | AGPL-3.0 HF / GPL-3.0 repo conflict | Owner/legal decision |
| M-LongDoc | Public repo | No LICENSE located | Benchmark sealed; RL denied until approval |
| Jina baselines | HF public | CC BY-NC 4.0 | Research-only baseline |
| DeepSeek-OCR-2 / Qwen / ColQwen2 | HF public | Apache-2.0 | Allowed subject to pinned terms |


## Appendix F. Where benchmark contamination could sneak in

**Appendix guide.** These are the main ways training data could overlap with or influence evaluation data.

| Risk | Source group | Evidence | Control |
| --- | --- | --- | --- |
| High | Visual-CoT parents; DUDE derivatives; BoundingDocs; TAT-QA/TAT-DQA; ChartQA/RefChartQA; MultiModalQA/M3DocVQA | Exact known derivative relationship | Canonical source ownership + exclusions |
| High | MMDocIR/MMDocRAG/MMLongBench/LongDocURL/M-LongDoc | Public long-doc sources may share PDFs | PDF/page hash quarantine |
| High | QASPER/PeerQA/SPIQA/SciEGQA/DocScope | Shared scientific papers/arXiv | DOI/arXiv/PDF hash |
| Medium | FinQA/MultiHiertt/DocFinQA/TAT family | Shared financial reports and 2,119 known questions | Question/report mapping |
| Medium | Wikipedia sources across HotpotQA/MultiModalQA/QuoteSum | Shared articles/entities/revisions | Title/revision/text hash |
| Medium | Generated templates within source | Near-identical questions/entities | Template clusters and document caps |


## Appendix G. Which source provides which fields

**Appendix guide.** A quick map of which native fields each source actually provides.

| Source | Question/answer | Original visual | Native boxes | Complete spatial | Text evidence | Program | Subgoals | Document IDs | Candidate mapping | License tag | Strength |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Visual-CoT | N | Y | Y | Y | N | N | N | Y | Y | Y | C |
| BoundingDocs | N | Y | Y | Y | N | N | N | Y | Y | Y | C |
| DUDE | Y | Y | Y | Partial | N | N | N | Y | Y | Y | A/B/C |
| TAT-DQA | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | A/B |
| QASPER | Y | After reacquisition | Derived | N | Y | N | N | Y | Derived | Y | B/C |
| PeerQA | Y | After reacquisition | Derived | N | Y | N | N | Y | Derived | Y | B/C |
| RefChartQA | N | Y | Y | Y | N | Partial | N | Y | Y | Y | A/C |
| FinQA | Y | No native | N | N | Y | Y | Y | Y | N | Y | B planner |
| TAT-QA | Y | No native | N | N | Y | Y | Y | Y | N | Y | B planner |
| MultiHiertt | Y | No native | N | N | Y | Y | Y | Y | N | Y | B planner |
| ConditionalQA | Y | HTML/text | N | N | Y | N | N | Y | N | Y | B planner |
| QuoteSum | Y | N | N | N | Y | N | N | Y | N | Y | B answerer |
| SPIQA | Y | Figures/tables | Reference asset | N | Y | Partial | N | Y | Partial | Y | B/D answerer |
| Sealed benchmarks | Varies | Y | Varies | Varies | Varies | Varies | Varies | Y | Varies | Y | Evaluation only |


## Appendix H. Real cited question examples from included sources

**Appendix guide.** These are authentic questions from official source material, not invented illustrations.

| Included source | Authentic questions | Primary citation |
| --- | --- | --- |
| Visual-CoT DocVQA | “what is the contact person name mentioned in letter?”; “Which corporation's letterhead is this?” | [VCOT-GH] |
| Visual-CoT InfographicVQA | “How many car crashes are caused by texting every year?”; “How many victims come from low- and middle-income communities?” | [VCOT-GH] |
| Visual-CoT SROIE | “What is the name of the restaurant?”; “What is the address of the restaurant?” | [VCOT-GH] |
| DUDE | “What ratio was used for Payout to Reported Household Income?”; “What was the total of securities transactions settlements including interest?” | [DUDE-P] |
| TAT-DQA | “What was the total cost in Wireless including spectrum license fee in 2019?”; second exact official row requires acquisition | [TATDQA-P] |
| BoundingDocs | “What is the total due after the current charges?”; “What name is written on the invoice?” | [BOUND-P] |
| QASPER | “What datasets are used to evaluate?”; “Is the model computed efficiently?” | [QASPER-P] |
| PeerQA | “Could supervised agents outperform the reported performance?”; “Why is memorization not effective?” | [PEERQA-P] |
| RefChartQA | “Referring to plotly purple, return the category with the lowest value.”; “What is the percentage of the third bar from the right?” | [REFCHART-P] |
| FinQA | “What was the percentage change in the research and development expenses from 2018 to 2019?”; “What was the average employee cost during FY18 and FY19?” | [FINQA-P] |
| TAT-QA | “How many years show an effective tax rate of 30% or higher?”; “What is the average effective tax rate from 2012 to 2018?” | [TATQA-P] |
| MultiHiertt | “What portion of total identifiable net assets is in cash?”; “Which segment had the most funds in 2017?” | [MULTIHIERTT-P] |
| ConditionalQA | “Can I get Housing Benefit to pay my second home?”; “Can I claim universal credit if I get child tax credit?” | [COND-GH] |
| QuoteSum | “what is the title of tears for fears song?”; “how long did it take to build the temple of jerusalem?” | [QUOTESUM-GH] |
| MultiModalQA | “When was the player who was drafted by the Memphis Grizzlies born?”; “What was the population of the country whose flag is shown?” | [MMQA-P] |
| HotpotQA | “Which magazine was started first, Arthur's Magazine or First for Women?”; “What government position was held by the woman who portrayed Corliss Archer?” | [HOTPOT-P] |
| BREAK | “Who is the father of the wife of the current president of the US?”; “What are the names of works by Ayn Rand that were adapted to film?” | [BREAK-P] |

Where a second exact example was unavailable from accessible official materials, the audit card says so explicitly. No constructed illustration is presented as an authentic dataset row.


## Appendix I. Datasets we rejected and the exact reason for each decision

**Appendix guide.** Every rejected source appears here with the decisive reason for rejection.

| Dataset/view | Decisive reason |
| --- | --- |
| DocVQA direct | Duplicate parent; no complete spatial evidence |
| InfographicVQA direct | Duplicate parent; no complete spatial evidence |
| SROIE direct | No native QA |
| TextVQA and Visual-CoT TextVQA | Scene-text domain, no native complete evidence, owner-excluded |
| TextCaps | Captioning, scene-domain, synthetic QA, overlap |
| Visual-CoT DUDE | Duplicate of original DUDE |
| BBox DocVQA v1 | Superseded identity |
| SciEGQA train | No explicit license; automatic labels unaudited |
| GroundingDocQA | No released assets/license |
| MESAQA | Unclear license and synthetic text-only redundancy |
| ChartQA | RefChartQA is stronger grounded derivative |
| MMDocIR train | Opaque mixture and severe overlap |
| “MultiDocVQA” | No canonical source |
| DocTrace | Method, not dataset |


## Appendix J. A final check that the numbers balance

**Appendix guide.** Final arithmetic checks confirm that the planned totals reconcile.

The report-generation checks assert:

```text
minimum_original = 18,000
minimum_accepted_augmentation = 14,000
minimum_selector_total = 32,000
minimum_generation_candidates = 32,500

full_original = 64,000
full_accepted_augmentation = 64,000
full_selector_total = 128,000
full_generation_candidates = 145,000
visual_cot_docvqa_original = 16,000
visual_cot_docvqa_accepted_augmentation = 9,600
textvqa_future_corpus_uses = 0
textcaps_future_corpus_uses = 0
planner_pool = 32,000
answerer_pool = 16,000
conditional_rl_pool = 20,000
full_future_program_total = 196,000
```

Any released manifest that differs from a target must carry a versioned amendment explaining which deterministic backfill rule was invoked. A failed quota remains a failed quota; it is never concealed by relabeling an estimate as an observed count.


[VCOT-P]: https://arxiv.org/pdf/2403.16999
[VCOT-GH]: https://github.com/deepcs233/Visual-CoT
[VCOT-HF]: https://huggingface.co/datasets/deepcs233/Visual-CoT
[DOCVQA-P]: https://arxiv.org/pdf/2007.00398
[DOCVQA-SITE]: https://rrc.cvc.uab.es/?ch=17
[INFO-P]: https://arxiv.org/pdf/2104.12756
[INFO-SITE]: https://rrc.cvc.uab.es/?ch=17
[SROIE-P]: https://arxiv.org/pdf/2103.10213
[SROIE-SITE]: https://rrc.cvc.uab.es/?ch=13
[TEXTVQA-P]: https://arxiv.org/pdf/1904.08920
[TEXTVQA-SITE]: https://textvqa.org/
[TEXTCAPS-P]: https://arxiv.org/pdf/2003.12462
[TEXTCAPS-SITE]: https://textvqa.org/textcaps/
[BOUND-P]: https://arxiv.org/pdf/2501.03403
[BOUND-HF]: https://huggingface.co/datasets/letxbe/BoundingDocs
[BBOX-V1]: https://arxiv.org/pdf/2511.15090v1
[SCIEGQA-P]: https://arxiv.org/pdf/2511.15090
[SCIEGQA-PROJ]: https://yuwenhan07.github.io/SciEGQA-project/
[SCIEGQA-GH]: https://github.com/yuwenhan07/SciEGQA
[SCIEGQA-TRAIN]: https://huggingface.co/datasets/Yuwh07/SciEGQA-Train
[SCIEGQA-BENCH]: https://huggingface.co/datasets/Yuwh07/SciEGQA-Bench
[M3G-PROJ]: https://m3grounder.github.io/
[M3G-P]: https://openaccess.thecvf.com/content/CVPR2026/html/Venna_M3Grounder_Mask-Based_Multi-Span_and_Multi-Granular_Grounding_for_Document_QA_CVPR_2026_paper.html
[MPDOC-P]: https://arxiv.org/pdf/2212.05935
[MPDOC-GH]: https://github.com/rubenpt91/MP-DocVQA-Framework
[MPDOC-HF]: https://huggingface.co/datasets/naver-clova-ix/MP-DocVQA
[DUDE-P]: https://arxiv.org/pdf/2305.08455
[DUDE-GH]: https://github.com/duchallenge-team/dude
[DUDE-HF]: https://huggingface.co/datasets/jordyvl/DUDE_loader
[SLIDE-P]: https://arxiv.org/pdf/2301.04883
[SLIDE-GH]: https://github.com/nttmdlab-nlp/SlideVQA
[SLIDE-LIC]: https://github.com/nttmdlab-nlp/SlideVQA/blob/main/LICENSE
[QASPER-P]: https://aclanthology.org/2021.naacl-main.365.pdf
[QASPER-HF]: https://huggingface.co/datasets/allenai/qasper
[FINQA-P]: https://aclanthology.org/2021.emnlp-main.300.pdf
[FINQA-GH]: https://github.com/czyssrs/FinQA
[TATQA-P]: https://aclanthology.org/2021.acl-long.254.pdf
[TATQA-GH]: https://github.com/NExTplusplus/TAT-QA
[MULTIHIERTT-P]: https://arxiv.org/pdf/2206.01347
[MULTIHIERTT-GH]: https://github.com/psunlpgroup/MultiHiertt
[COND-P]: https://arxiv.org/pdf/2110.06884
[COND-GH]: https://github.com/haitian-sun/ConditionalQA
[MESAQA-P]: https://aclanthology.org/2025.coling-main.724.pdf
[MESAQA-GH]: https://github.com/reiiwang/MESAQA
[MESAQA-HF]: https://huggingface.co/datasets/riiwang/MESAQA
[QUOTESUM-P]: https://arxiv.org/pdf/2311.04886
[QUOTESUM-GH]: https://github.com/google-research-datasets/QuoteSum
[MLONG-P]: https://arxiv.org/pdf/2411.06176
[MLONG-GH]: https://github.com/kenchan0226/multimodal-docs-public
[MMQA-P]: https://arxiv.org/pdf/2104.06039
[MMQA-GH]: https://github.com/allenai/multimodalqa
[HOTPOT-P]: https://arxiv.org/pdf/1809.09600
[HOTPOT-SITE]: https://hotpotqa.github.io/
[BREAK-P]: https://arxiv.org/pdf/2001.11770
[BREAK-GH]: https://allenai.org/data/break
[TATDQA-P]: https://arxiv.org/pdf/2207.11871
[TATDQA-GH]: https://github.com/NExTplusplus/TAT-DQA
[TATDQA-HF]: https://huggingface.co/datasets/next-tat/TAT-DQA
[PEERQA-P]: https://aclanthology.org/2025.naacl-long.22.pdf
[PEERQA-GH]: https://github.com/UKPLab/PeerQA
[REFCHART-P]: https://arxiv.org/pdf/2503.23131
[REFCHART-GH]: https://github.com/moured/RefChartQA
[REFCHART-HF]: https://huggingface.co/datasets/omoured/RefChartQA
[SPIQA-P]: https://arxiv.org/pdf/2407.09413
[SPIQA-HF]: https://huggingface.co/datasets/google/spiqa
[HITAB-P]: https://aclanthology.org/2022.acl-long.78.pdf
[HITAB-GH]: https://github.com/microsoft/HiTab
[CHARTQA-P]: https://aclanthology.org/2022.findings-acl.177.pdf
[CHARTQA-GH]: https://github.com/vis-nlp/ChartQA
[LONGDOCURL-P]: https://aclanthology.org/2025.acl-long.57.pdf
[LONGDOCURL-PROJ]: https://longdocurl.github.io/
[M3DOC-P]: https://arxiv.org/pdf/2411.04952
[M3DOC-PROJ]: https://m3docrag.github.io/
[MMDOCIR-P]: https://arxiv.org/pdf/2501.08828
[MMDOCIR-GH]: https://github.com/texttron/MMDocIR
[MMDORAG-P]: https://arxiv.org/pdf/2505.16470
[MMDORAG-GH]: https://github.com/Kudux/MMDocRAG
[MMLBD-P]: https://arxiv.org/pdf/2404.15754
[MMLBD-GH]: https://github.com/yubo-ma/MMLongBench-Doc
[MMLBD2-P]: https://arxiv.org/abs/2608.03397
[MMLBD2-GH]: https://github.com/meiu-zhang/MMlongBench-Doc
[MMLBD2-HF]: https://huggingface.co/datasets/meiu/MMlongBench-Doc
[DOCSCOPE-P]: https://arxiv.org/abs/2605.08888
[DOCSCOPE-PROJ]: https://mililab.github.io/DocScope/
[CHARTQAPRO-P]: https://vis-nlp.github.io/ChartQAPro/
[CHARTQAPRO-PROJ]: https://vis-nlp.github.io/ChartQAPro/
[DOCFINQA-P]: https://gaspar.ai/resources/docfinqa/
[DOCFINQA-GH]: https://github.com/ibm/DocFinQA
[DOCTRACE-P]: https://arxiv.org/pdf/2608.03292
[DOCRE-P]: https://arxiv.org/pdf/2505.22584
[DOCRE-PROJ]: https://navvewas.github.io/DocReRank/
[DSOCR2-HF]: https://huggingface.co/deepseek-ai/DeepSeek-OCR-2
[DSOCR2-P]: https://arxiv.org/pdf/2601.20552
[QWEN-RERANK-HF]: https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B
[QWEN-RERANK-P]: https://arxiv.org/pdf/2601.04720
[QWEN-EMB-HF]: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
[JINA-EMB-HF]: https://huggingface.co/jinaai/jina-embeddings-v5-omni-small-retrieval
[JINA-M0-HF]: https://huggingface.co/jinaai/jina-reranker-m0
[COLQWEN-HF]: https://huggingface.co/vidore/colqwen2-v1.0
