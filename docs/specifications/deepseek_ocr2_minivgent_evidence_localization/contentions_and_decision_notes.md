# Contentions and Decision Notes for the Evidence-Localization Corpus

**Status:** Nonbinding research and decision record
**Recorded:** August 7, 2026
**Scope boundary:** This note remains nonbinding and does not itself authorize dataset acquisition, training, model execution, or expansion of the active overlap-first V1 answer-anchor POC.

## Current owner decision: first BoundingDocs document-QA corpus

**Decision date:** 2026-08-07

The first BoundingDocs corpus is now limited to the three sources that retain
human-written document questions:

| Source | Documents | Pages | Questions |
|---|---:|---:|---:|
| DUDE | 2,583 | 13,832 | 4,512 |
| MP-DocVQA | 5,203 | 57,643 | 31,597 |
| SP-DocVQA | 266 | 266 | 419 |
| **Selected union** | **8,052** | **71,741** | **36,528** |

Acquire the pinned BoundingDocs v2.0 release and preserve its document-level
train/validation/test assignments. A later first training run may sample
15,000–20,000 questions from the resulting training pool; acquisition itself
retains every eligible record. Kleister Charity and Kleister NDA remain useful
long-document sources for later teacher-generated questions, but they are not
part of this first corpus.

This decision supersedes the broader all-source BoundingDocs starter proposed
later in this note. The earlier discussion remains below as decision history
and technical background. The prepared SOL acquisition handoff is
[`../../../sol/archive/source-repo/task_spec/boundingdocs_document_qa_acquisition_handoff.md`](../../../sol/archive/source-repo/task_spec/boundingdocs_document_qa_acquisition_handoff.md).

## Why this file exists

This file records places where the corpus design is contested, where the owner's intuition changes the proposed direction, and where the research assessment agrees only with qualifications. It is intentionally separate from the main specification so that disagreements remain visible instead of being silently edited away.

## Contention 1: Should BoundingDocs or Visual-CoT come first?

### Owner position

BoundingDocs should be used first because it is large, already includes OCR, already includes answer locations, and appears ready for immediate training.

### Research assessment

**Agree with the main decision:** BoundingDocs is the stronger first-stage bootstrap dataset. The earlier Visual-CoT-first recommendation should be reversed.

BoundingDocs already packages:

- page images;
- Amazon Textract OCR with word and line geometry;
- questions and rephrased questions;
- answer page identifiers and answer-word boxes;
- 48,151 documents and 249,016 question-answer pairs; and
- document-level training, validation, and test splits.

This removes the need to run OCR or manually draw answer boxes before the first training run. The corrected `v2.0` branch should be pinned at commit `dd2c05ae4d516bff617409ceb942d49f22452852`.

### Pushback: what “immediately trainable” should and should not mean

BoundingDocs is close to trainable, but it is not a zero-preparation dataset and it does not completely match the final evidence-selection task.

1. **It primarily marks answer text, not every supporting region.** Its boxes are excellent supervision for “find the answer-bearing words or page.” They may omit table headers, legends, qualifiers, calculation operands, and other regions needed to prove the answer.

2. **Its sources are highly imbalanced.** FATURA supplies 102,403 questions—about 41% of all BoundingDocs questions—and DeepForm supplies another 55,926. Naive question-level sampling would allow a few document families to dominate training.

3. **It overlaps other planned sources.** BoundingDocs contains DUDE, MP-DocVQA, and SP-DocVQA records. Those records must be removed or tracked by parent document when the original datasets are used elsewhere.

4. **Its coordinates require normalization.** BoundingDocs answer boxes use a `0–1000` coordinate system, while its Textract OCR boxes use `0–1`. The loader must convert both into one canonical representation and test the conversion visually.

5. **Its OCR and QA fields are serialized records.** A training adapter must parse the Textract and QA JSON, construct candidate regions, and connect each answer box to its candidate words or lines.

6. **Its parent-source rights must remain visible.** BoundingDocs is marked CC BY 4.0, but it repackages eleven sources. Acquisition should retain parent provenance and verify the terms applicable to each source.

### Recommended sequencing

1. **BoundingDocs first:** bootstrap question-conditioned answer-anchor and page localization.
2. **Visual-CoT second:** provide a matched comparison and additional document styles after source filtering and deduplication.
3. **DUDE and TAT-DQA later:** add genuine multi-page, unanswerable, list, calculation, and broader evidence requirements.
4. **Verified synthetic examples after that:** add complete evidence sets, hard negatives, and controlled cross-page composition.
5. **TextVQA and TextCaps remain excluded.**

The defensible label for BoundingDocs is therefore **answer-anchor localization supervision**, not **complete-evidence supervision**. BoundingDocs can teach the model where an answer string appears without proving that those boxes contain everything needed to justify the answer.

## What are FATURA and DeepForm?

They are two of the eleven parent sources bundled inside BoundingDocs. Neither began as a natural question-answering dataset. BoundingDocs converted their labeled fields into questions and localized the answers using Textract OCR.

### FATURA

FATURA is a **synthetic invoice-image dataset**.

- It contains 10,000 one-page invoices generated from 50 layouts.
- Its layouts were designed from the structural blueprints of selected real invoices, but the released invoices use generated content.
- Names, addresses, product descriptions, amounts, and other values are filled with randomly selected or generated text.
- Each layout receives a generated logo, and the invoice generator checks relationships such as whether the total matches the product amounts.
- The original annotations cover 24 invoice regions or fields, including invoice ID, purchase date, seller, buyer, tax, subtotal, total, amount due, payment details, and purchase-order number.
- BoundingDocs converts those fields into 102,403 QA records—an average of 10.24 questions per invoice.

**Why it is useful:** It provides clean boxes, plentiful examples, many invoice fields, and controlled layout variation. It is well suited to an early localization warm-up.

**Why it should be capped:** Its 10,000 images come from only 50 synthetic templates. Training heavily on every QA can teach template recognition and repeated field extraction more than general document evidence reasoning. Evaluation should hold out entire templates, not merely new invoices made from templates seen during training.

### DeepForm

DeepForm is a collection of **real, public, multi-page political-advertising documents** obtained from U.S. Federal Communications Commission public files. The documents include advertising orders, contracts, invoices, and disclosure forms used to report television or cable political-ad spending.

The original task is narrow key-information extraction rather than open-ended QA. It targets five field types:

- advertiser;
- contract number;
- flight-from date;
- flight-to date; and
- gross amount.

For BoundingDocs, the authors downloaded the source PDFs, ran Amazon Textract over them, matched the labeled values to OCR words, and converted the field names into questions such as “What is the gross amount?” The questions were also given language-model rewrites to reduce the repetition of the basic templates.

Within BoundingDocs, DeepForm contributes:

- 24,345 documents;
- 100,747 pages; and
- 55,926 questions, averaging 2.30 questions per document.

**Why it is useful:** These are real, often multi-page documents with difficult tables, forms, and layout variation. DeepForm is more valuable than FATURA for learning robustness to real document noise and page structure.

**Why it is limited:** Almost every training question asks for one of only five fields. The wording may vary, but the underlying information need remains narrow. It teaches field localization well; it does not supply broad question diversity or complete multi-evidence reasoning.

### Do not confuse DeepForm with VRDU Ad-buy Forms

BoundingDocs lists **DeepForm** and **VRDU Ad Form** as separate sources. They concern the same general political-advertising domain and draw from the same FCC resource, but VRDU Ad-buy Forms were independently collected and annotated with a richer schema, including repeated and hierarchical line-item fields. They must remain separate in manifests and source-level reporting.

## Practical source policy

For a BoundingDocs-first bootstrap:

- use DeepForm as real-document answer-location training, but cap its five repeated question families;
- use FATURA as synthetic invoice warm-up data, but cap its total share and group splits by template;
- sample by source and document rather than allowing datasets with many fields per page to dominate;
- preserve original and rephrased questions as separate provenance fields;
- exclude overlapping DUDE, MP-DocVQA, and SP-DocVQA documents before any claimed independent evaluation; and
- do not treat either FATURA or DeepForm as complete-evidence supervision.

## Contention 2: Should the BoundingDocs sources be merged from the beginning?

### Owner position

Because BoundingDocs already combines several useful datasets, the first training corpus should probably merge them instead of treating each parent source as an isolated training stage.

### Research assessment

**Agree, with an important boundary:** Begin with a merged, source-balanced BoundingDocs corpus. Do not begin with a blind merger of BoundingDocs and Visual-CoT.

BoundingDocs is already a unified merger of eleven parent datasets. Separately fine-tuning on FATURA, then DeepForm, then each remaining source would encourage sequential overfitting to narrow document families. Mixing the eligible BoundingDocs sources from the beginning is the better starting point.

However, “merged” must mean **one common training interface with preserved provenance**, not one flattened pool in which all questions are treated as equivalent.

### All eleven sources inside BoundingDocs

The following are the records that BoundingDocs retained after its own filtering and OCR matching. These are not necessarily the complete original dataset totals.

| BoundingDocs source | Plain-language description | Documents | Questions | Main contribution | Main limitation |
|---|---|---:|---:|---|---|
| DeepForm | Real FCC political-advertising orders, contracts, and invoices | 24,345 | 55,926 | Real multi-page forms and difficult tables | Only five repeated field types |
| FATURA | Synthetic invoices generated from 50 layouts | 10,000 | 102,403 | Clean invoice-field localization at scale | Synthetic templates and extreme question share |
| DUDE | Real, varied, multi-page documents with human-written questions | 2,583 | 4,512 | Natural questions and broad layouts | BoundingDocs keeps only OCR-localizable answers |
| FUNSD | Noisy, scanned forms, including handwriting | 199 | 1,542 | OCR noise and form relationships | Very small dataset |
| Kleister Charity | Long charity annual and financial reports | 2,169 | 8,897 | Long-document navigation and financial fields | Narrow predetermined field families |
| Kleister NDA | Legal nondisclosure agreements | 337 | 696 | Parties, dates, terms, and legal layouts | Small and narrowly structured |
| MP-DocVQA | Complete multi-page industry documents with human questions | 5,203 | 31,597 | Finding the relevant page and answer | Overlaps DocVQA material used elsewhere |
| SP-DocVQA | Single document pages with human questions | 266 | 419 | General single-page QA | Most overlapping pages were already removed through MP-DocVQA |
| VRDU Ad-buy Forms | Real FCC political-ad purchasing forms | 641 | 22,506 | Dense tables, repeated entries, and hierarchical fields | About 35 questions per document can dominate question-level sampling |
| VRDU Registration Forms | U.S. foreign-agent registration forms | 1,015 | 3,865 | Government forms, registrants, foreign principals, and signatures | Few templates and field families |
| XFUND | Forms in seven non-English languages | 1,393 | 16,653 | Multilingual form localization | Parent-split leakage must be repaired before original-benchmark evaluation |

Together these sources produce 48,151 documents, 237,437 pages, and 249,016 question-answer pairs.

### What each remaining source actually adds

#### DUDE

Original DUDE is valuable because it contains real, varied, multi-page documents and human-written questions, including extractive, abstractive, list, and unanswerable cases.

The BoundingDocs version is narrower. BoundingDocs removes questions whose answers cannot be matched directly to OCR text. Its DUDE subset is therefore useful for answer localization, but it does not replace original DUDE's broader reasoning and abstention supervision.

#### FUNSD

FUNSD contains only 199 documents, but they are real, noisy scans with handwritten content and difficult layouts. Its original labels distinguish headers, questions, answers, and their relationships. BoundingDocs converts those relationships into ordinary localized questions.

FUNSD is too small to provide scale, but it helps prevent the model from assuming that all documents look as clean as synthetic invoices.

#### Kleister Charity

Kleister Charity contains long annual reports from charity organizations. Questions concern fields such as charity name, charity number, address, annual income, and annual spending.

It contributes genuine long-document structure, but its information needs still come from a small field schema rather than open-ended human questions.

#### Kleister NDA

Kleister NDA contains nondisclosure agreements and confidentiality contracts. Its fields include parties, effective date, jurisdiction, and agreement term.

It adds legal-document layouts and vocabulary, but its BoundingDocs contribution is only 337 documents and 696 questions.

#### MP-DocVQA

MP-DocVQA places questions inside their complete multi-page documents. It is particularly valuable for teaching the model to identify the relevant page before locating the answer.

Its questions are human-written, unlike the questions generated from information-extraction field names. However, its documents can overlap SP-DocVQA, Visual-CoT's DocVQA sources, and separate DocVQA experiments. Document-level deduplication is mandatory.

#### SP-DocVQA

SP-DocVQA asks human-written questions about individual document pages. BoundingDocs retains only 266 documents and 419 questions because it removed most pages already represented through MP-DocVQA.

This residual subset adds little unique scale and should be excluded whenever independent DocVQA or Visual-CoT comparisons cannot be made leakage-safe.

#### VRDU Ad-buy Forms

VRDU Ad-buy Forms contain political-advertising purchases with dense tables and repeated line items. They have a richer schema than DeepForm, including fields tied to particular advertising programs.

This is useful structured localization data, but it contributes 22,506 questions from only 641 documents. Without document-balanced sampling, those few forms would receive disproportionate training influence.

#### VRDU Registration Forms

VRDU Registration Forms are public U.S. Foreign Agents Registration Act documents. Their fields include registrant name, registration number, foreign principal, signer name, and signer title.

They add another real government-form domain, although the layouts and requested fields remain relatively constrained.

#### XFUND

XFUND provides form-understanding examples in Chinese, Japanese, Spanish, French, Italian, German, and Portuguese. It is the main multilingual component of BoundingDocs.

There is an evaluation-leakage complication: the BoundingDocs paper reports that 313 original FUNSD/XFUND test documents were placed in the BoundingDocs training split. Those documents must be identified and removed before reporting independent results on the original FUNSD or XFUND test sets.

### The merge that should happen

The first training corpus should use the eligible records from the BoundingDocs `v2.0` training split through one unified manifest. Every record must retain at least:

- parent dataset;
- parent document identifier;
- BoundingDocs split and recoverable parent split;
- real-versus-synthetic document status;
- human-written, templated, or model-rephrased question provenance;
- answer page and answer boxes;
- OCR version and original coordinate system;
- overlap and deduplication status; and
- supervision label identifying it as answer-anchor rather than complete-evidence supervision.

### The merge that should not happen

The 249,016 questions should not simply be shuffled at their natural frequencies. That would make FATURA about 41% of the corpus and DeepForm about 22%, while dense VRDU Ad-buy documents would receive approximately 35 times as many training opportunities as documents with one question.

The first sampler should instead:

1. choose a parent source using controlled source weights;
2. choose a document within that source;
3. choose one eligible question from that document; and
4. rotate through that document's remaining questions over later epochs.

This preserves all useful questions while preventing datasets with many labeled fields per document from dominating the model.

### Why Visual-CoT should not enter the first merged run

Visual-CoT should remain a second controlled experiment. If BoundingDocs and Visual-CoT are combined immediately, it becomes difficult to determine whether Visual-CoT added useful question or document diversity, merely duplicated DocVQA material, or introduced weaker pseudo-localization.

The recommended experimental sequence is:

1. **BoundingDocs balanced mixture:** train on the leakage-safe, source-balanced BoundingDocs union.
2. **BoundingDocs plus Visual-CoT:** add filtered, deduplicated Visual-CoT and measure its marginal contribution.
3. **Original DUDE and TAT-DQA curriculum:** add unanswerable, list, calculation, and broader multi-evidence behavior.
4. **Verified synthetic evidence sets:** add complete-evidence, hard-negative, and controlled cross-page supervision that the real datasets do not provide.

The resulting decision is:

> Start with all safe BoundingDocs sources merged through a provenance-preserving, source-balanced loader. Do not merge Visual-CoT into the first run, and do not treat the official BoundingDocs question distribution as an appropriate training distribution.

## Primary sources

- [BoundingDocs dataset card](https://huggingface.co/datasets/letxbe/BoundingDocs)
- [BoundingDocs paper](https://arxiv.org/abs/2501.03403)
- [FATURA paper](https://arxiv.org/abs/2311.11856)
- [FATURA release](https://zenodo.org/records/8261508)
- [DUDE paper](https://arxiv.org/abs/2305.08455)
- [FUNSD paper](https://arxiv.org/abs/1905.13538)
- [Kleister Charity and NDA paper](https://arxiv.org/abs/2105.05796)
- [DocVQA dataset page](https://site.docvqa.org/datasets/docvqa)
- [MP-DocVQA paper](https://arxiv.org/abs/2212.05935)
- [XFUND and LayoutXLM paper](https://arxiv.org/abs/2104.08836)
- [VRDU repository and source descriptions](https://github.com/google-research-datasets/vrdu)
- [VRDU paper](https://arxiv.org/abs/2211.15421)
