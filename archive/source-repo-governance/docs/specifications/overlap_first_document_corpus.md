# Overlap-First Document Corpus

**Status:** Current corpus design. Source acquisition and the full physical
overlap scan are still in progress.

**Decision date:** 2026-08-07

## The plain-English decision

We will build one master document collection from datasets that reuse the same
underlying documents. Repeated copies of a question will become one canonical
question. Useful annotations from every copy—answer boxes, page labels, OCR,
answers, and retrieval negatives—will be retained on that one record.

This is an **overlap-aware union**, not a strict intersection. A strict
intersection would keep only questions found in every dataset and would throw
away most of the useful data.

The initial collection has three document families:

1. the DocVQA / MP-DocVQA family;
2. the DUDE family; and
3. the InfographicVQA family.

The first pipeline run will sample 15,000–20,000 high-confidence localization
examples from this collection. That pilot is smaller than the collection; it
does not redefine it.

## What is merged first

| Document family | Sources treated as different views of the same documents |
|---|---|
| DocVQA | Original SP-DocVQA, original MP-DocVQA, Visual-CoT DocVQA, BoundingDocs MP-DocVQA and SP-DocVQA, and MMDocIR MP-DocVQA |
| DUDE | Original DUDE, Visual-CoT DUDE, BoundingDocs DUDE, and MMDocIR DUDE |
| InfographicVQA | Original InfographicVQA and Visual-CoT InfographicVQA |

Original datasets supply document and question identity. Visual-CoT and
BoundingDocs supply answer-region boxes. BoundingDocs also supplies Textract
OCR. MMDocIR supplies positive pages and mined negative pages. These are
complementary annotations, not independent examples when the document and
question are the same.

Forms, receipts, invoices, scene photographs, and multilingual sources remain
outside this collection. This excludes TextVQA, TextCaps, SROIE, FATURA,
DeepForm, FUNSD, VRDU forms, and XFUND. Kleister Charity and Kleister NDA remain
possible later document families but are not part of the first collection.

## What the release metadata already proves

The following counts describe released rows before cross-source merging. They
must not be added together as if every row were unique.

| Source view | Released rows or questions | Parent documents observed or reported |
|---|---:|---:|
| Visual-CoT DocVQA | 33,453 | 5,131 recovered from image-name prefixes |
| Visual-CoT DUDE | 11,735 | 2,904 recovered from document hashes in image names |
| Visual-CoT InfographicVQA | 15,055 | 3,805 image-documents |
| BoundingDocs MP-DocVQA | 31,597 | 5,203 |
| BoundingDocs DUDE | 4,512 | 2,583 published; the materialized manifest controls |
| BoundingDocs SP-DocVQA | 419 | 266 |
| MMDocIR MP-DocVQA | 6,122 | 512 observed in its annotation file |
| MMDocIR DUDE | 2,715 | 767 observed in its annotation file |

The released identifiers are strong enough for direct joins:

- Visual-CoT DocVQA image `xnbl0037_1.png`, MP-DocVQA document `xnbl0037`,
  and the corresponding BoundingDocs `doc_id` share the parent identifier.
- Visual-CoT DUDE image names begin with the original DUDE document hash;
  BoundingDocs and MMDocIR retain that hash.
- InfographicVQA image IDs remain visible in Visual-CoT filenames.

Image hashes are still required to catch renamed, recompressed, or incorrectly
linked pages.

## Overlap measured so far

These are metadata-level measurements using normalized question text plus the
recovered parent-document identifier. They are not estimates.

| Comparison | Shared parent documents | Exact normalized document-question pairs |
|---|---:|---:|
| Visual-CoT DocVQA ↔ MMDocIR MP-DocVQA | 490 | 4,207 |
| Visual-CoT DUDE ↔ MMDocIR DUDE | 611 | 2,111 |

Of MMDocIR's canonicalized pairs, 73.1% of MP-DocVQA and 79.6% of DUDE are
already present in Visual-CoT. MMDocIR should therefore contribute retrieval
annotations to existing records whenever possible, not duplicate training
weight.

The earlier BoundingDocs-to-Visual-CoT metadata audit found 8,152
BoundingDocs document-question pairs not already present in Visual-CoT:

- 7,164 MP-DocVQA pairs;
- 917 DUDE pairs; and
- 71 residual SP-DocVQA pairs.

Before incorporating MMDocIR, the current box-bearing union is therefore
66,440 canonical document-question pairs. MMDocIR can raise that total by no
more than 2,092 additional pairs, producing a current bound of **66,440 to
68,532**. The final count remains pending because the new MMDocIR pairs must be
checked against the 8,152 BoundingDocs additions and all conflicts and bad
records must be quarantined.

Original MP-DocVQA, DUDE, and InfographicVQA contain additional questions that
may not have accepted localization boxes. They belong in the master corpus but
are not automatically eligible for localization SFT.

## How one canonical record works

One record represents one question asked about one canonical parent document.
It contains:

- a canonical document ID and ordered page IDs;
- the original question and a normalized copy used only for matching;
- every source question ID and original split;
- answers and answer type;
- every compatible answer box, with its source and coordinate system;
- positive answer or evidence pages;
- MMDocIR negative pages when available;
- every OCR version, linked rather than silently overwritten;
- a conflict and review status; and
- task eligibility such as localization SFT, page-retrieval SFT, answerer
  training, teacher-scored RL, or evaluation only.

The same document with a different question remains a separate record. The
same generic question on a different document also remains separate.

## Duplicate and conflict rules

1. Join first by native parent identity and source question ID when available.
2. Confirm pages with exact file hashes and deterministic rendered-page hashes.
3. Use perceptual image hashes and OCR similarity to find renamed or
   recompressed copies.
4. Collapse only the same parent document plus the same normalized question.
5. Retain all compatible annotations on the canonical record.
6. Quarantine records when duplicated questions disagree about the answer,
   answer page, or evidence box. Do not silently select one source.
7. Never delete a source record from the audit ledger; mark it as merged,
   unique, conflicting, excluded, or evaluation-only.

## Training views of the same corpus

The master corpus is not one flat training file.

| Training view | Records allowed |
|---|---|
| Localization SFT | Questions with accepted answer/evidence boxes |
| Page-retrieval SFT | Questions with accepted positive pages and audited negatives |
| Answer training | Questions with a trusted reference answer, even when boxes are incomplete |
| Teacher-scored RL | Training-document questions for which answer correctness and evidence sufficiency can be scored reliably |

An original question without boxes can still be useful for answer training or
RL. It must not be mislabeled as supervised localization data.

## Splits are created after merging

Every acquired training-authorized record is merged before our internal split
is assigned. The canonical parent document—not the individual page or
question—is assigned to 90% training, 5% development, or 5% internal test.
Every derivative and every annotation view follows that document.

Official test documents and sealed external benchmarks remain outside this
process. They may enter the overlap scanner only to generate a contamination
alert. They cannot supply training questions, teacher prompts, negatives, or
threshold decisions. Using official test records in our new split would make
later official benchmark claims invalid.

## Expansion through additional overlap families

We will expand by acquiring another document family together with as many of
its compatible annotation views as possible. The known candidates are:

| Expansion family | Known or potential overlap | Decision |
|---|---|---|
| TAT financial reports | TAT-QA ↔ TAT-DQA ↔ MMDocIR TAT-DQA; TAT-DQA explicitly extends TAT-QA | Highest-priority expansion for numerical and table-plus-text reasoning |
| Slide decks | SlideVQA ↔ MMDocIR SlideVQA | Useful multi-page visual family; add after the document pilot |
| Financial reports | FinQA ↔ DocFinQA for 7,437 questions; MultiHiertt imports 2,119 FinQA questions; report-level overlap with TAT must be checked | Add only after report IDs and PDFs are reconciled |
| Scientific papers | MMDocIR ArxivQA and SciQAG; possible shared papers among QASPER, PeerQA, SPIQA, SciEGQA, and DocScope | DOI, arXiv ID, title, and PDF-hash audit required before use |
| Charts | ChartQA ↔ RefChartQA directly; ChartQAPro may reuse chart sources | Separate chart-focused expansion, not silently mixed into the document pilot |
| Wikipedia documents | MultiModalQA ↔ M3DocVQA directly; possible overlap with MMDocIR Wiki-ss and HotpotQA | Keep separate until stable page-revision identity is available |
| Long public PDFs | Possible overlap among LongDocURL, M-LongDoc, MMDocRAG, MMLongBench-Doc, and DocScope | Evaluation quarantine first; training use requires a clean identity audit |
| GroundingDocQA | Parent-corpus overlap has not been verified from a complete public manifest | Do not merge until the release and lineage are auditable |

MMDocIR's released training annotations already expose seven parent views:
MP-DocVQA, DUDE, TAT-DQA, SlideVQA, ArxivQA, SciQAG, and Wiki-ss. Its compact
JSONL annotations can be joined before downloading its roughly 49 GB of page
Parquet files. This prevents paying storage and OCR costs for pages already in
the master corpus.

## What happens next

1. Finish the current Visual-CoT and BoundingDocs acquisitions without changing
   their source snapshots.
2. Acquire original parent manifests and the compact MMDocIR annotation files.
3. Build the native-ID overlap ledger before copying or OCR-processing more
   images.
4. Run exact and perceptual page matching across every candidate document.
5. Publish final unique, merged, conflict, and excluded counts by family.
6. Freeze the 90/5/5 document groups.
7. Draw the 15,000–20,000 localization pilot from training documents only.

## Pinned sources used in this decision

- [Visual-CoT dataset](https://huggingface.co/datasets/deepcs233/Visual-CoT/tree/223d2d8c1146fda2bb918801b8276c587b78b61c), commit `223d2d8c1146fda2bb918801b8276c587b78b61c`.
- [BoundingDocs v2.0](https://huggingface.co/datasets/letxbe/BoundingDocs/tree/v2.0), commit `dd2c05ae4d516bff617409ceb942d49f22452852`.
- [MMDocIR training data](https://huggingface.co/datasets/MMDocIR/MMDocIR_Train_Dataset/tree/059e7a30e87429698eaead28c30b1613dcf915c8), commit `059e7a30e87429698eaead28c30b1613dcf915c8`.
- [MP-DocVQA paper](https://arxiv.org/abs/2212.05935) and [official framework](https://github.com/rubenpt91/MP-DocVQA-Framework).
- [DUDE paper](https://arxiv.org/abs/2305.08455) and [official repository](https://github.com/duchallenge-team/dude).
- [InfographicVQA paper](https://arxiv.org/abs/2104.12756).
- [MMDocIR paper and repository](https://github.com/mmdocrag/MMDocIR).
- [TAT-DQA paper](https://arxiv.org/abs/2207.11871).
- [DocFinQA paper](https://arxiv.org/abs/2401.06915).
- [MultiHiertt paper](https://aclanthology.org/2022.acl-long.454/).
- [M3DocVQA construction repository](https://github.com/bloomberg/m3docrag/tree/main/m3docvqa).
- [RefChartQA paper](https://arxiv.org/abs/2503.23131).

