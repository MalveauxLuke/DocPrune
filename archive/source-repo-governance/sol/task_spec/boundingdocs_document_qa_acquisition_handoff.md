# BoundingDocs Document-QA Bootstrap: Prepared SOL Handoff

**Status:** Owner-approved active SOL handoff. The SOL agent must first rebind
the repository's active routing from Evidence-DINO-Units to BoundingDocs, then
download the dataset directly to the dedicated scratch directory.

**Decision date:** 2026-08-07

**Immediate goal:** Acquire and audit the BoundingDocs v2.0 records whose
source is exactly `DUDE`, `MP-DocVQA`, or `SP-DocVQA`, while preserving the
release's document-level train/validation/test split and every provenance
field needed for later evidence-localization training.

This handoff is intentionally limited to acquisition, source selection,
schema validation, and audit-ready manifests. It does not authorize training,
question generation, RL, model downloads, or adding other BoundingDocs
sources.

## 1. Authority and activation boundary

Read these files before acting:

1. root `AGENTS.md`;
2. `docs/SOL_INSTRUCTIONS.md`;
3. `sol/AGENTS.md`;
4. `docs/EXPERIMENT_WORKSPACES.md`;
5. `agent-context/INDEX.md`; and
6. this handoff.

The owner explicitly supersedes every statement in the required-read files
that still calls Evidence-DINO-Units the active task. Those statements are
stale routing, not a blocker. The SOL agent must remove them through the
control-plane transition below. Preserve the prior project's specifications,
branches, worktrees, manifests, and archived task record; do not delete or
reuse them.

### 1.1 Required SOL control-plane transition

Perform this transition in the clean control checkout before downloading data:

1. Fetch `origin`, fast-forward local `main`, and record the verified local and
   remote commit. Stop if `main` is dirty, diverged, or cannot fast-forward.
2. Update root `AGENTS.md` and `sol/AGENTS.md` so BoundingDocs is the active SOL
   task and this handoff is its binding specification. Remove the active
   Evidence-DINO branch, worktree, scratch, and stage-boundary instructions
   from both files.
3. Update `agent-context/CURRENT_TASK.md`, `agent-context/INDEX.md`,
   `docs/ACTIVE_PROJECTS.md`, `docs/EXPERIMENT_WORKSPACES.md`, `README.md`, and
   `sol/CURRENT_SOL_TASK.md` so they consistently identify BoundingDocs as
   active. Record that it uses the existing control checkout plus the scratch
   root from Section 1.2, with no separate worktree or branch.
   Evidence-DINO-Units may remain only as paused or archived history.
4. Update the existing routing-contract tests, especially
   `tests/evidence_dino/test_handoff.py`. Replace assertions that force
   Evidence-DINO content into current-task files with assertions for this
   BoundingDocs handoff. Do not satisfy old tests by retaining stale text in an
   active prompt.
5. Preserve all Evidence-DINO specifications, branches, worktrees, data, and
   archives. This is a routing change, not deletion or cleanup.
6. Run only the checks proportional to this routing edit:
   `git diff --check` and
   `python -m pytest -q tests/test_documentation_contract.py tests/evidence_dino/test_handoff.py --tb=short`.
   Require zero failures. Do not run the unrelated full repository suite for
   this documentation transition or for dataset downloading. Broaden testing
   only if the agent changes executable code outside the acquisition tooling,
   or a focused failure requires a wider diagnostic run.
7. Commit only the routing files and corresponding tests, push the
   control-plane commit to `origin/main`, fetch again, and require local `HEAD`
   to equal `origin/main`. Do not add a co-author line and do not include local
   scratch content.

This small documentation-and-test transition and any small acquisition
scripts, manifests, or reports are authorized on `main`. Do not create a
separate worktree or branch for this download task.

### 1.2 Required scratch setup

Use only these locations:

```text
control checkout: ~/COLPALI_binary_classification on main
scratch root:     /scratch/$USER/boundingdocs_document_qa
```

Require at least 300 GiB free at the approved scratch root before any download.
All Parquet files, caches, temporary data, and heavy audit outputs belong under
that scratch root. Only small scripts, manifests, and reports may be committed
to the control checkout. Do not download dataset files into home or Git.

Completing Sections 1.1 and 1.2 is not a handback point. Continue directly
into the acquisition and audit procedure unless a stated hard stop is hit.

## 2. Frozen corpus decision

Select only these exact, case-sensitive `source` values:

```text
DUDE
MP-DocVQA
SP-DocVQA
```

Exclude every other BoundingDocs source from the first corpus, including
DeepForm, FATURA, FUNSD, Kleister Charity, Kleister NDA, both VRDU sources, and
XFUND. Kleister remains a possible later source for teacher-generated
questions, but it is not part of this acquisition deliverable.

Use the release's existing `train`, `validation`, and `test` assignments. Do
not merge those splits and do not create a new random split. Sampling a later
15,000–20,000-record first run is a training-plan decision, not an acquisition
filter.

## 3. Pinned release

```text
Hub repository: letxbe/BoundingDocs
release:        v2.0
commit:         dd2c05ae4d516bff617409ceb942d49f22452852
access date:    2026-08-07
visibility:     public and ungated
card license:   CC BY 4.0
format:         Parquet
```

Use the immutable commit, not `main` and not the mutable `v2.0` tag, in every
download and manifest.

Version 2.0 is required because it fixes MP-DocVQA page ordering between
`doc_images` and `doc_ocr` and standardizes `rephrased_question`. The default
release described by the card remains v1.0 and must not be used.

Primary sources:

- [BoundingDocs v2.0 release](https://huggingface.co/datasets/letxbe/BoundingDocs/tree/v2.0)
- [BoundingDocs paper](https://doi.org/10.1007/s10032-025-00563-5)
- [MP-DocVQA paper](https://arxiv.org/abs/2212.05935)
- [DUDE paper](https://arxiv.org/abs/2305.08455)

The wrapper card states CC BY 4.0, but the selected documents inherit parent
provenance and data-use constraints. Treat the corpus as research-only. Do not
redistribute images, annotations, derived datasets, or trained weights until a
parent-rights matrix is approved.

## 4. Verified release facts

### 4.1 Physical release

The pinned v2.0 release contains:

```text
488 repository files total
486 Parquet files
389 train Parquet files
48 validation Parquet files
49 test Parquet files
190,580,291,346 download bytes (190.58 GB; about 177.49 GiB)
242,165,878,132 decoded dataset bytes reported by the card
```

The full release has 48,151 document rows:

```text
train documents:      38,515
validation documents:  4,804
test documents:        4,832
```

One row is one document, not one question. Questions are nested inside the
row's `Q&A` string.

### 4.2 Selected-source totals across all three release splits

| Source | Documents | Pages | Questions |
|---|---:|---:|---:|
| DUDE | 2,583 | 13,832 | 4,512 |
| MP-DocVQA | 5,203 | 57,643 | 31,597 |
| SP-DocVQA | 266 | 266 | 419 |
| **Selected union** | **8,052** | **71,741** | **36,528** |

These are expected release-wide totals, not expected training-split totals.
The acquisition audit must calculate and record exact document, page,
question, answer-instance, and box counts for each `(split, source)` pair.

BoundingDocs assigned documents source-by-source to an 80/10/10 split and
kept all questions for a document together. It also removed SP-DocVQA pages
already represented in MP-DocVQA. Preserve those choices, but still run an
identity audit across the three selected sources.

### 4.3 Supervision supplied

The three selected sources retain their human-written questions. BoundingDocs
did not create template questions for them. BoundingDocs did filter out
abstractive DUDE and DocVQA questions whose answers could not be aligned to OCR
text. This corpus therefore teaches question-conditioned answer-anchor
localization. It does not reproduce the full DUDE task and does not guarantee
complete supporting evidence.

## 5. Actual row schema and parser rules

Each document row has exactly these top-level fields:

```text
source: string
doc_id: string
doc_images: sequence<image>
doc_ocr: sequence<string>
Q&A: string containing JSON
```

After `json.loads(row["Q&A"])`, the result is an object keyed by QA ID, not a
list. Iterate it as:

```python
for qa_id, qa in json.loads(row["Q&A"]).items():
    question = qa["question"]
    rephrased_question = qa["rephrased_question"]
    answers = qa["answers"]
```

Do not copy the release card's `qa_data[0]` example; that example does not
match the released object structure.

Each answer object contains:

```text
value: answer text
location: list of word-level boxes
page: 1-based page number
```

Important coordinate issue:

- answer boxes are documented as `[width, height, x, y]` on a 0–1000 scale;
- Textract OCR boxes use a 0–1 scale; and
- the paper describes `(x, y)` as a top-left point while the v2.0 card calls it
  a bottom-left point.

Do not resolve that contradiction by assumption. Determine the actual origin
with rendered overlays before writing a canonical-coordinate converter.

Multiple answer objects and multiple boxes inside one answer must remain
separate. Never replace them with one enclosing rectangle.

## 6. Known quality and leakage risks

1. **Answer anchors are not complete evidence.** A localized answer string may
   omit headers, table labels, comparison values, or other necessary context.
2. **Repeated-string false positives exist.** BoundingDocs matched answer text
   to Textract output and retained all occurrences. The paper estimates that
   about 7% of all annotations may contain an unrelated repeated occurrence.
3. **MP page alignment requires v2.0.** Reject any materialized record if image
   count, OCR-page count, or page ordering is inconsistent.
4. **SP and MP share parent provenance.** BoundingDocs reports removing known
   SP pages already in MP, but the local audit must still compare document IDs,
   normalized image hashes, and OCR fingerprints.
5. **Future dataset overlap is guaranteed.** If original DUDE, DocVQA,
   Visual-CoT DocVQA, or Visual-CoT DUDE is used later, join by parent identity
   before selecting any train or evaluation record.
6. **The release split is not an external benchmark.** The BoundingDocs test
   split may be used for an internal pipeline check, but publication claims
   require separately quarantined external evaluation data.

## 7. SOL acquisition procedure

### 7.1 Preflight on the login node

After the routing transition is pushed, verify that the control checkout is on
clean, synchronized `main`. Also record:

```bash
df -h "$BOUNDINGDOCS_SCRATCH_ROOT"
df -i "$BOUNDINGDOCS_SCRATCH_ROOT"
hf version
hf datasets info letxbe/BoundingDocs \
  --revision dd2c05ae4d516bff617409ceb942d49f22452852
```

Set `BOUNDINGDOCS_SCRATCH_ROOT` exactly to
`/scratch/$USER/boundingdocs_document_qa`; reject an empty or different value.

Require at least 300 GiB free before submitting the download. The release is
about 177.49 GiB compressed and about 225.53 GiB decoded; the additional space
protects resumability and audit outputs without assuming a second full copy.

Do not install packages or download data on the login node.

### 7.2 Dry run inside a CPU allocation

Use a CPU allocation suitable for sustained network and filesystem work. Set
the approved scratch locations and keep the Hugging Face cache on scratch.
Supply `HF_TOKEN` through the environment for rate limits; never place it in a
command, log, manifest, or Git file.

Run:

```bash
hf download letxbe/BoundingDocs \
  --repo-type dataset \
  --revision dd2c05ae4d516bff617409ceb942d49f22452852 \
  --dry-run \
  --format json
```

The dry run must resolve 488 files and 190,580,291,346 bytes. Record the JSON
and command version in the stage report before downloading.

### 7.3 Download

Download the pinned snapshot to the approved scratch raw-data directory:

```bash
hf download letxbe/BoundingDocs \
  --repo-type dataset \
  --revision dd2c05ae4d516bff617409ceb942d49f22452852 \
  --local-dir "$BOUNDINGDOCS_RAW" \
  --max-workers 8
```

The task must define `BOUNDINGDOCS_RAW` from the approved scratch root before
running this command. Do not point it at home, the control checkout, or another
project's scratch tree.

Do not create a second copy of the full Parquet release merely to filter three
sources. Keep the downloaded snapshot immutable and build small manifests that
reference its relative shard paths and row identities.

### 7.4 Verify the snapshot

Run the Hub verifier against the local snapshot:

```bash
hf cache verify letxbe/BoundingDocs \
  --type dataset \
  --revision dd2c05ae4d516bff617409ceb942d49f22452852 \
  --local-dir "$BOUNDINGDOCS_RAW" \
  --fail-on-missing-files \
  --fail-on-extra-files
```

Also write a file inventory containing relative path, byte size, and Hub LFS
SHA-256 for every downloaded file. A successful job exit without this
verification is not an acquisition pass.

## 8. Required manifests and reports

All large Parquet files and rendered overlays stay on scratch. The control
checkout receives only small, reviewable outputs:

```text
data/manifests/boundingdocs_release.json
data/manifests/boundingdocs_files.jsonl
data/manifests/boundingdocs_selected_documents.jsonl
data/manifests/boundingdocs_selected_questions.jsonl
data/manifests/boundingdocs_quarantine.jsonl
reports/boundingdocs/acquisition.md
reports/boundingdocs/schema_audit.md
reports/boundingdocs/coordinate_audit.md
```

Every selected-question row must retain at least:

```text
release_commit
boundingdocs_split
source
doc_id
qa_id
question
rephrased_question
answers
relative_parquet_path
document_row_index
page_count
```

Use the stable record ID:

```text
boundingdocs-v2::<split>::<source>::<doc_id>::<qa_id>
```

Do not put absolute SOL paths in committed manifests.

## 9. Audit gates

### Gate A — release integrity

- pinned commit is exactly
  `dd2c05ae4d516bff617409ceb942d49f22452852`;
- 488 files and 190,580,291,346 download bytes are accounted for;
- Hub verification passes; and
- no dataset file exists outside the approved scratch root.

### Gate B — selected corpus

- only the three exact source labels are selected;
- release-wide expected totals reconcile to 8,052 documents, 71,741 pages,
  and 36,528 questions;
- exact per-split counts are observed and recorded;
- every QA ID is unique within its document and every stable record ID is
  globally unique; and
- train, validation, and test document identities are disjoint.

If v2.0 does not reproduce the published release-wide totals, stop and report
the exact observed difference rather than forcing the manifest to match the
paper.

### Gate C — schema and page alignment

- every `Q&A` value parses as a JSON object;
- every selected QA contains a nonempty question and at least one answer;
- each answer page is within `1..len(doc_images)`;
- `len(doc_images) == len(doc_ocr)` for every document; and
- a stratified overlay audit establishes the true box origin and verifies MP
  image/OCR ordering.

Render 120 QA overlays: 40 per source, spread across available splits, with
priority given to multi-page, multiple-answer, repeated-number, and
multiple-box cases. Record every failure with a reason code. Do not silently
repair or discard it.

### Gate D — duplicate and false-positive audit

- compare normalized image hashes and OCR fingerprints across the three
  sources and splits;
- quarantine any cross-split visual duplicate;
- flag repeated-answer occurrences that do not answer the question; and
- report acceptance and quarantine counts by source and split.

## 10. Stop point and handback

Stop after the pinned snapshot, selected manifests, audits, observed counts,
and quarantine report are complete and the small Git-safe outputs are
committed and pushed to `main`.

Return:

- commit;
- SOL job IDs;
- physical snapshot location;
- exact source/split counts;
- verification evidence;
- overlay-audit results;
- duplicate/quarantine results;
- license status; and
- the next safe action.

Do not train a model, generate new questions, create an RL corpus, add Kleister
or Visual-CoT, or change the 15,000–20,000 first-run sampling plan during this
handoff.
