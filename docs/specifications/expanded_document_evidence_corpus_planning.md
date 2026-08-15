# Planning the Expanded Document-Evidence Corpus

**Status:** Early planning document; not an acquisition or training authority.

**Starting point:**
[`overlap_first_document_corpus.md`](overlap_first_document_corpus.md)

**Research background:** the detailed source evidence and earlier alternatives
remain in
[`deepseek_ocr2_minivgent_evidence_localization/contentions_and_decision_notes.md`](deepseek_ocr2_minivgent_evidence_localization/contentions_and_decision_notes.md)
and the
[`reader-friendly corpus guide`](deepseek_ocr2_minivgent_evidence_localization/dataset_specification_friendly_guide.md).

## Why a second corpus is needed

The overlap-first corpus combines the DocVQA, DUDE, and InfographicVQA
document families while collapsing repeated document-question pairs. It uses
Visual-CoT, BoundingDocs, original parent manifests, and MMDocIR as annotation
views rather than treating every released row as an independent example.

The next corpus should add capabilities that answer boxes alone do not teach:

- finding several pieces of evidence together;
- using labels, headers, and surrounding context;
- reasoning across pages;
- handling list, numerical, abstractive, and unanswerable questions;
- deciding when the selected evidence is sufficient; and
- learning from a teacher model without requiring manual boxes for every new
  question.

## Proposed expansion sequence

### 1. Materialize the overlap-first corpus

Combine document-focused Visual-CoT with the BoundingDocs DUDE, MP-DocVQA, and
SP-DocVQA intake. Join original parent manifests and MMDocIR's DUDE and
MP-DocVQA annotations. Collapse matching document-question pairs while keeping
all compatible boxes, OCR, answers, positive pages, and audited negatives.

Do not add TextVQA, TextCaps, SROIE, or receipt-oriented records.

Create new 90/5/5 parent-document splits only after this merge. Draw the first
15,000–20,000 localization pilot from the merged training documents.

### 2. Add the TAT overlap family

Combine TAT-QA, TAT-DQA, and MMDocIR TAT-DQA by report and question identity.
This is the highest-priority expansion for numerical and table-plus-text
reasoning. Also retain original DUDE questions that add list, abstractive,
multi-page, or unanswerable behavior missing from box-bearing derivatives.

Parent-document identity must be resolved before these records enter training
so the same document cannot cross training and evaluation boundaries.

### 3. Add other overlap families

Next candidates are SlideVQA with MMDocIR, FinQA with DocFinQA and MultiHiertt,
and scientific-paper sources joined by DOI, arXiv ID, and PDF hash. Sealed
benchmarks remain evaluation-only even when an overlap is detected.

### 4. Generate better questions for long documents

Use Kleister Charity and Kleister NDA as possible long-report and legal-document
sources. Their template field questions are not sufficient by themselves.
Instead, a teacher model can propose natural questions that require useful
document evidence.

Generated questions must be checked for answerability, evidence grounding,
duplication, and leakage before they become supervised examples.

### 5. Create a separate RL pool

RL data should remain separate from supervised localization data. It can use
documents and teacher-generated questions without manually drawn boxes when a
teacher or deterministic checker can score:

- whether the answer is correct;
- whether the selected evidence contains the answer;
- whether the evidence is sufficient;
- whether irrelevant regions were selected; and
- whether the model correctly abstains on unanswerable questions.

Teacher output is a proposed reward signal, not automatically trusted ground
truth. A verified reward-quality audit is required before RL begins.

## Sources currently outside the plan

- TextVQA and TextCaps;
- SROIE and other receipt datasets;
- DeepForm, FATURA, FUNSD, and VRDU form datasets;
- multilingual XFUND;
- sealed external evaluation datasets; and
- any source whose license or parent-document overlap is unresolved.

## Counts to decide after the initial download

We will not invent precise expansion counts before measuring:

1. the accepted BoundingDocs train count;
2. source and question-type balance;
3. duplicate overlap with Visual-CoT and original DUDE;
4. TAT-DQA conversion yield;
5. teacher-question acceptance rate; and
6. how many RL prompts produce reliable rewards.

The next planning pass will turn those measurements into a table showing, for
each training stage, every contributing dataset, original count, generated
count, supervision type, and reason for inclusion.

## Decisions already fixed

- The BoundingDocs source intake remains unchanged, but it is no longer the
  complete first training corpus.
- The first corpus is the overlap-aware union defined in
  `overlap_first_document_corpus.md`.
- New sources are added in visible stages, not silently merged.
- Dataset names and counts remain separate even when stages overlap.
- TextVQA, TextCaps, receipts, forms, and multilingual data remain excluded.
- Supervised examples and RL episodes remain separate datasets.
- External evaluation data remains sealed from training and teacher prompts.
