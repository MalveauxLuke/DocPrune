# COLPALI Research Architecture Registry

This registry tracks nonbinding research paths for question-conditioned
evidence localization in document images. It separates reviewed source papers,
project-specific architecture proposals, and approved or executed experiment
records.

> **Nonbinding research context:** A registry entry does not change an active
> task, authorize implementation, or establish that a paper's results transfer
> to COLPALI. Adoption requires project-owner approval and a separate update to
> the relevant canonical project specification.

## Current research snapshot

The current default proposal direction retains fixed, inspectable DeepSeek-OCR
semantic units and establishes an independent document-specific multimodal
reranker as the simplest semantic baseline. Query rewriting broadens the
training distribution through intent-preserving positive variants, while
DocReRank-style meaning-altered rewrites provide verified hard negatives.
Mini-VGent is a later set-aware treatment for complementarity, redundancy, and
multi-unit evidence. DeepSeek-OCR-2 generative localization is a separate
alternative for testing whether document-specific pretraining can overcome the
recall ceiling of fixed candidates. VGOQ contributes global page-relevance and
no-evidence ideas.

Current registry state:

- three reviewed and promising paper records: VGent, VGOQ, and HierDoc;
- six project proposal records, including one adopted for the active
  overlap-first V1 baseline;
- one nonbinding future corpus specification for controlled DeepSeek-OCR2 and
  MiniVGent evidence-localization experiments, with a reader-friendly edition;
- one mini-VGent adaptation retained inside the VGent paper record; and
- no approved, scheduled, running, or completed architecture-registry
  experiment.

The active overlap-first V1 reranker specification—not this registry—now
authorizes a narrow stock baseline and same-model hard-negative fine-tuning
sequence after its workspace gate closes. Nothing in this registry expands
that phase to MiniVGent, residual-candidate training, DeepSeek-OCR2
fine-tuning, synthetic multi-hop, answer-feedback RL, or another architecture
change.

## Research program map

```text
Training-data layer
├── Query rewriting
│   └── Wider positive-query coverage
└── Negative contrastive query rewriting
    └── Similar but unanswerable hard-negative queries

Fixed-candidate model path
├── DeepSeek-OCR semantic units
├── Document-specific multimodal reranker   <- simplest semantic baseline
├── HierDoc region-ID policy                <- later sufficiency baseline
└── Mini-VGent set-aware decoder            <- later treatment
    └── Residual-image candidate            <- proposed candidate-miss escape

Alternative localization path
└── DeepSeek-OCR-2 generative boxes         <- tests fixed-candidate ceiling

Auxiliary paper mechanisms
├── VGOQ global page/no-evidence head
└── VGent multi-candidate interaction
```

## Scientific decision logic

This is a dependency and interpretation order, not an approved execution
schedule:

1. Measure fixed-unit oracle coverage before attributing failures to a
   selector.
2. If coverage is high, establish the independent document-specific
   multimodal reranker as the simplest semantic baseline.
3. Test positive query augmentation and verified query-side hard negatives as
   training-data changes with candidates and model architecture fixed.
4. Test mini-VGent only if independent scoring fails specifically on
   complementary, redundant, relational, or multi-unit evidence.
5. Treat DeepSeek-OCR-2 generative localization as a separate branch if fixed
   candidates impose a meaningful recall ceiling.
6. Evaluate VGOQ-style page relevance, no-evidence prediction, and abstention
   separately from unit ranking.

## Reviewed paper ledger

| Paper | Actual task and domain | Reusable mechanism | Possible COLPALI consumers | Applicability | Primary promise | Primary limitation | Evidence | Status |
|---|---|---|---|---|---|---|---|---|
| [VGent](papers/vgent-2512.11099.md) (`2512.11099v1`) | Natural-image single- and multi-target visual grounding | Frozen MLLM hidden-state memory plus a set-aware decoder over detector proposals | Evidence candidate/unit selection; multi-region set reasoning | `architectural_analogue` | Coordinated selection over high-recall proposals without autoregressive coordinate generation | No document evidence task, OCR/layout representation, or answer-sufficiency evaluation | Method: `paper_reported`; COLPALI fit: `repository_inference` | `promising` |
| [VGOQ](papers/vgoq-cvpr-2026.md) (CVPR 2026) | Natural/product-image grounding of evidence or context for general object questions | Separate spatial evidence heatmap and global image-relevance head over frozen CLIP features | Evidence-DINO-Units page/no-evidence head; semantic grounding evaluation; future scene supervision | `adjacent` | Directly tests question-to-evidence semantics rather than only named referring expressions | Synthetic data, single-image masks, and no document structure or complete evidence sets | Method: `paper_reported`; COLPALI fit: `repository_inference` | `promising` |
| [HierDoc](papers/hierdoc-2607.29638.md) (`2607.29638v1`) | Multi-page and long-document VQA with explicit page and region evidence routing | Separate GRPO-trained page-ID and parser-region-ID set policies; selected pages plus local crops/OCR for answering | Later answer-sufficiency baseline; future multi-page routing and evidence-conditioned answering | `direct` | Closest external system to question-conditioned selection of compact semantic document-region sets | V1 supplies answer anchors rather than complete evidence sets; the paper's 8B generative policy is not compute matched to MiniVGent | Method: `paper_reported`; COLPALI fit: `repository_inference` | `promising` |

## Project proposal ledger

| Proposal | Role in program | Core treatment | Decisive reason to test it | Main limitation | Status |
|---|---|---|---|---|---|
| [DeepSeek-OCR-2 generative evidence localization](proposals/deepseek_ocr2_generative_localization.md) | Alternative localization branch | Condition DeepSeek-OCR-2 on the question and generate supporting boxes instead of Markdown | Tests whether document-specific pretraining can localize implicit evidence beyond phrase grounding or a fixed candidate universe | Autoregressive boxes can be malformed, incomplete, duplicated, or oversized | `unvalidated_proposal` |
| [Document-specific multimodal reranker](proposals/document_specific_multimodal_reranker.md) | Simplest semantic model baseline | Independently score each question-unit pair with a pretrained 2B multimodal reranker, optionally followed by task post-training | Strong low-complexity semantic baseline with no coordinate generation | Independent scores do not naturally model complementarity, redundancy, or set competition | `adopted_for_overlap_v1_baseline` |
| [MiniVGent answer-anchor POC](proposals/minivgent_answer_anchor_poc.md) | Source proposal for active POC and later sufficiency test | Active POC compares pairwise Qwen with MiniVGent without/with interaction; later work retains HierDoc and matched A1 | Isolates pairwise adaptation, shared semantic memory, and candidate interaction before introducing structured sufficiency policies | V1 labels answer anchors rather than complete evidence; HierDoc needs stronger sufficiency supervision | `partially_adopted` |
| [MiniVGent residual-image candidate](proposals/minivgent_residual_candidate.md) | Future fixed-candidate fallback | Add one bbox-free candidate containing the masked page complement outside all ordinary candidate regions | Softens candidate misses for downstream answering without requiring raw coordinate generation | Broad fallback can become a shortcut and does not recover a precise box | `unvalidated_proposal` |
| [Query rewriting](proposals/query_rewriting.md) | Positive training-data augmentation | Expand each authoritative training query into a wider family of intent-preserving positive rewrites | Broadens training-query coverage and reduces reliance on exact wording | Positive rewrites can drift in intent or cease to share the same sufficient evidence | `unvalidated_proposal` |
| [Negative contrastive query rewriting](proposals/negative_contrastive_query_rewriting.md) | Hard-negative training-data augmentation | Following DocReRank, hold evidence fixed and generate similar but meaning-altered queries as candidate hard negatives | May teach rerankers to distinguish semantic correctness from vocabulary overlap | Generated negatives require strong answerability verification to avoid false-negative supervision | `unvalidated_proposal` |

The reviewed [VGent paper record](papers/vgent-2512.11099.md) remains the source
for MiniVGent's decoder inspiration. The reviewed
[HierDoc paper record](papers/hierdoc-2607.29638.md) is the source for the later
answer-sufficiency page-to-region experiment. The separate
[MiniVGent answer-anchor POC](proposals/minivgent_answer_anchor_poc.md) records
the owner-approved, dataset-specific controlled comparison. Its
[Qwen implementation-readiness companion](proposals/minivgent_qwen_implementation_readiness.md)
freezes the proposed hidden-state, visual-grid, candidate-tensor, decoder,
loss, and systems contracts needed by a future implementation agent. Neither
file changes the active experiment or authorizes execution.

## Nonbinding corpus specifications

- [Controlled DeepSeek-OCR2 and MiniVGent dataset specification](../../../docs/specifications/deepseek_ocr2_minivgent_evidence_localization/dataset_specification.md):
  the full source-audited corpus, augmentation, supervision, fairness,
  evaluation, and allocation proposal.
- [Friendly research guide](../../../docs/specifications/deepseek_ocr2_minivgent_evidence_localization/dataset_specification_friendly_guide.md):
  a reader-oriented edition that preserves the complete specification while
  adding plain-language labels, guideposts, and dataset-card summaries.
- [Contentions and decision notes](../../../docs/specifications/deepseek_ocr2_minivgent_evidence_localization/contentions_and_decision_notes.md):
  a separate record of the BoundingDocs-first proposal, the limits of its
  answer-anchor supervision, and plain-language explanations of FATURA and
  DeepForm.

These files remain research provenance. The approved single-hop pairwise-Qwen
and MiniVGent POC, plus the deferred HierDoc/A1 answer-sufficiency design, have
been consolidated into the binding
`docs/specifications/overlap_v1_evidence_localization/` design family.
DeepSeek-OCR2 fine-tuning, page-policy training, synthetic multi-hop,
answer-feedback RL, complete-evidence claims, and unrelated acquisition remain
outside that program.

## Ongoing experiments

See [experiments/README.md](experiments/README.md) for the experiment-record
contract. The
[overlap-first V1 evidence-localization POC](experiments/overlap_v1_evidence_localization.md)
is approved for staged execution but is not scheduled or running. The former
[segment-reranker record](experiments/overlap_v1_segment_reranker_baseline.md)
is retained as its R0/R1 predecessor.

## How to use this registry

1. Start with the research program map and scientific decision logic.
2. Use the paper or proposal ledger to locate the relevant mechanism.
3. Open only the detailed records needed for the current question.
4. Check the applicability, evidence level, and status before reusing a claim.
5. Treat proposed COLPALI integrations as hypotheses until tested locally.
6. Add a paper file only after checking enough of its primary source to support
   every factual statement in the record.
7. Keep architecture hypotheses under `proposals/`; create an `experiments/`
   record only after a concrete experiment is approved.

## Capability categories

Entries may use more than one category:

- page or document retrieval;
- candidate or proposal generation;
- question-conditioned representation;
- region, unit, or patch selection;
- multi-region and set reasoning;
- semantic and structural closure;
- answer sufficiency and necessity;
- supervision and pseudo-labeling;
- abstention, recovery, and uncertainty;
- parsing and document representation;
- answer generation and grounded attribution;
- token, latency, and memory efficiency; and
- evaluation and calibration.

## Applicability

- `direct`: the paper evaluates substantially the same task and domain.
- `adjacent`: the paper evaluates a related document or multimodal task.
- `architectural_analogue`: a mechanism may transfer, but the evaluated task
  or domain is materially different.

## Evidence levels

- `paper_reported`: directly stated or measured in the reviewed primary
  source.
- `repository_inference`: a reasoned mapping from the paper to COLPALI.
- `unvalidated_proposal`: a change or experiment that has not been implemented
  or run.
- `repository_tested`: implemented and measured locally with retained
  evidence.

One record may use several evidence levels. Each project-facing claim should
make its level clear.

## Registry statuses

- `unreviewed`: identified, but its complete primary source has not been
  reviewed.
- `reviewed`: methodology and evidence have been checked against the primary
  source.
- `promising`: reviewed and connected to a concrete COLPALI opportunity.
- `scheduled`: an approved controlled experiment exists.
- `tested`: the experiment ran and retained evidence exists.
- `adopted`: the relevant canonical project specification was explicitly
  approved and updated.
- `rejected`: a documented result or incompatibility rules out the proposal.
- `superseded`: a later entry or architecture replaces it.

Registry presence alone never means `adopted`.

## Adoption boundary

Before a registry idea affects implementation:

1. Verify its task, mechanism, training, and evidence from primary sources.
2. Name the COLPALI component it would replace or augment.
3. State a falsifiable expected advantage and failure interpretation.
4. Specify a controlled comparison.
5. Obtain project-owner approval.
6. Update the relevant canonical project specification separately.

## Adding records

Copy [TEMPLATE.md](TEMPLATE.md) only after checking the paper's primary source.
Link to the paper and keep its factual capsule short; use the file primarily
for our analysis, interpretation, open questions, and proposed experiments.
Do not create shallow placeholder files for papers that have not been reviewed.
Name files using a stable short name and arXiv ID when available, for example
`paper-name-2512.11099.md`.
