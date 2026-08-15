# BoundingDocs Intake for the Overlap-First Document Corpus

**Status:** Frozen source-intake definition; materialization is pending the SOL
download and audit. This file no longer defines the complete first training
corpus. The current merged-corpus decision is
[`overlap_first_document_corpus.md`](overlap_first_document_corpus.md).

**Decision date:** 2026-08-07

## The decision

The BoundingDocs intake is the audited BoundingDocs v2.0 data from exactly
three sources:

- `DUDE`
- `MP-DocVQA`
- `SP-DocVQA`

We will acquire every eligible record from those sources and preserve the
train, validation, and test assignments as source metadata.

This is the same source slice defined by the SOL acquisition handoff. The
handoff remains acquisition-only. After acquisition, these records enter the
overlap ledger with Visual-CoT, original parent manifests, and MMDocIR before
any merged training split is frozen.

## Expected size before the download audit

| Source | Documents | Pages | Human-written questions |
|---|---:|---:|---:|
| DUDE | 2,583 | 13,832 | 4,512 |
| MP-DocVQA | 5,203 | 57,643 | 31,597 |
| SP-DocVQA | 266 | 266 | 419 |
| **Complete selected corpus** | **8,052** | **71,741** | **36,528** |

These are release-wide totals across train, validation, and test. The final
manifest must report the observed counts for each `(split, source)` pair after
filtering and quarantine. Observed counts control if they differ from these
expected totals.

## What counts as training data

No record becomes a merged training example merely because it appears in this
intake. The final corpus split is assigned at canonical parent-document level
after cross-source merging. Original source splits remain attached for audit,
and official or sealed test documents remain ineligible for training.

The accepted intake contains all source questions. A smaller
15,000–20,000-question first run may be drawn from that pool to prove the
pipeline and model setup only after overlap reconciliation. That run-level
sample does not redefine the dataset. Its exact size and manifest must be
recorded before training begins.

## What one example contains

Each example preserves:

- the original human-written question;
- the optional BoundingDocs rephrased question as metadata, not as a second
  training example;
- the complete document and page order;
- page images and page OCR;
- every answer object;
- every word-level answer box;
- the answer page number;
- source, document, question, split, release, and shard provenance; and
- a stable ID of
  `boundingdocs-v2::<split>::<source>::<doc_id>::<qa_id>`.

Multiple answers and multiple boxes remain separate. They must not be replaced
by one enclosing rectangle.

## What the dataset teaches

This dataset teaches **question-conditioned answer-anchor localization**:
given a human question and a document, locate the page and OCR regions where
the answer text appears.

It does not guarantee that the annotated boxes contain every piece of evidence
needed to explain or derive the answer. Headers, table labels, comparison
values, and other context may be missing. We therefore do not call these boxes
complete evidence.

## What is not in this intake

- no Visual-CoT inside the downloaded BoundingDocs files; Visual-CoT is joined
  later through the overlap ledger;
- no TAT-DQA;
- no TextVQA or TextCaps;
- no SROIE, receipts, forms, invoices, or multilingual XFUND;
- no Kleister Charity or Kleister NDA yet;
- no generated questions;
- no synthetic evidence sets;
- no augmented copies;
- no answerer-only examples; and
- no RL episodes.

Those are possible later additions, not hidden parts of the initial corpus.

## Acceptance rules

A question is accepted only when:

1. its source is one of the three exact labels above;
2. its `Q&A` value parses correctly;
3. it has a nonempty question and at least one answer;
4. every answer page is valid for the document;
5. image and OCR page counts agree;
6. the document is not a cross-split visual duplicate;
7. its answer boxes survive the coordinate and overlay audit; and
8. its provenance and stable ID are complete.

Failures are quarantined with a reason. They are never silently repaired or
silently moved between splits.

## Frozen source

```text
Hub repository: letxbe/BoundingDocs
release:        v2.0
commit:         dd2c05ae4d516bff617409ceb942d49f22452852
format:         Parquet
```

The binding acquisition and audit procedure is
[`../../sol/archive/source-repo/task_spec/boundingdocs_document_qa_acquisition_handoff.md`](../../sol/archive/source-repo/task_spec/boundingdocs_document_qa_acquisition_handoff.md).

## What must exist before training starts

- the verified pinned snapshot on SOL scratch;
- observed source-by-split counts;
- accepted document and question manifests;
- quarantine and duplicate reports;
- coordinate-overlay audit results;
- a frozen training manifest containing only accepted `train` records; and
- separate frozen development and internal-test manifests.

No training is authorized merely by this specification. Training begins only
after those artifacts exist and a training task names the exact manifest it
will use.
