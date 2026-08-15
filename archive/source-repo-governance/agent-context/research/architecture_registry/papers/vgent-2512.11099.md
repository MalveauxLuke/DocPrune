# VGent: Visual Grounding via Modular Design for Disentangling Reasoning and Prediction

## Registry metadata

- Paper: VGent: Visual Grounding via Modular Design for Disentangling
  Reasoning and Prediction
- Stable identifier and reviewed version: CVPR 2026 proceedings version; arXiv
  `2512.11099v1`
- Last reviewed: 2026-08-04
- Applicability: `architectural_analogue`
- Capability tags: candidate or proposal generation; question-conditioned
  representation; region, unit, or patch selection; multi-region and set
  reasoning; abstention, recovery, and uncertainty
- Possible project consumers: Evidence-DINO-Units; hierarchical evidence
  routing; future segment rerankers
- Evidence levels used: `paper_reported`, `repository_inference`,
  `unvalidated_proposal`
- Registry status: `promising`

## Paper link and factual capsule

- Primary paper:
  [CVPR 2026 Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Kang_VGent_Visual_Grounding_via_Modular_Design_for_Disentangling_Reasoning_and_CVPR_2026_paper.html)
- Conference entry:
  [CVPR 2026 virtual poster](https://cvpr.thecvf.com/virtual/2026/poster/39372)
- Alternate paper entry:
  [arXiv abstract and PDF, `2512.11099v1`](https://arxiv.org/abs/2512.11099)

**Evidence level: `paper_reported`.**

VGent is a natural-image visual-grounding system, not a document-understanding
system. It uses detector boxes as proposals, projects them into proposal
queries, and selects matching objects through a decoder that cross-attends to
frozen MLLM hidden states and self-attends across proposals. The paper also
tests GRPO-based multi-target reasoning, mask-aware proposal labels, and global
count queries. Its experiments cover natural-image referring and segmentation
benchmarks; they do not demonstrate document evidence selection, OCR/layout
reasoning, answer sufficiency, or document-token savings.

Read the linked paper for full methodology, training settings, results, and
ablations.

## Why this paper attracted attention

**Evidence level: `repository_inference`.**

Our current Evidence-DINO-Units design directly scores fixed semantic
candidates. That is deliberately simple and scientifically useful, but it may
miss relationships among candidates: two weak units can jointly answer a
question, one unit can make another redundant, and several relevant units can
compete with one dominant answer-bearing region.

VGent provides a concrete architecture for separating:

- high-recall spatial proposals;
- semantic reasoning memory; and
- coordinated set selection.

That separation resembles our proposal-generator/selector boundary even though
VGent's actual targets are objects rather than supporting document evidence.

## Our analysis and interpretation

**Evidence level: `repository_inference`.**

VGent is most useful to us as a decoder pattern, not as a pretrained document
model. Its important idea is that proposal geometry can remain fixed while
proposal queries interact with a richer semantic memory and with one another.

This could address part of the evidence-grounding chicken-and-egg problem:
the high-recall candidate generator can nominate plausible locations without
knowing the answer, while a set-aware semantic decoder can reason about which
combination of candidates appears useful. It does not remove the circularity
around knowing whether the selected evidence is complete. Answer-sufficiency
evaluation, structural closure, abstention, and possible recovery would still
be required.

The paper's global count mechanism is suggestive but does not transfer
directly. Object count has comparatively clear ground truth. Evidence-set
cardinality depends on candidate granularity and may have several equally
valid decompositions. A document version may be better served by question-
obligation coverage or a sufficiency state than by a literal evidence count.

The paper also reinforces a useful experimental separation: proposal oracle
coverage and learned proposal selection should be reported independently.
A diverse proposal union can have an excellent ceiling while the selector
still fails.

## Reusable mechanisms

**Evidence level: `repository_inference`.**

1. Candidate queries cross-attending to frozen question-conditioned VLM
   hidden states.
2. Candidate self-attention for complementarity and redundancy.
3. Geometry supplied by proposal candidates rather than autoregressive box
   generation.
4. Multiple proposal sources combined before semantic selection.
5. Optional global queries that communicate set-level state to candidate
   queries; VGent's original object-count targets should not be assumed to
   transfer to document evidence.
6. Candidate-aware label construction that does not force one arbitrary
   spatial decomposition.

## Possible COLPALI integration points

**Evidence level: `repository_inference`.**

| Consumer | Current component | Possible reuse | Why it may help |
|---|---|---|---|
| Evidence-DINO-Units | Direct semantic-unit scorer | Add a small set-aware decoder over the same fixed candidates | Tests whether candidate interaction improves distributed evidence recall |
| Hierarchical evidence routing | Independent relevance decisions across branches | Maintain a global selected-set or coverage state | Could distinguish complementary branches from redundant branches |
| Future segment reranker | Independent or listwise scoring | Cross-attend candidate queries to question-conditioned VLM memory | Could improve implicit evidence judgment without regressing boxes |

None of these mappings has been implemented or measured.

## Required adaptations

**Evidence level: `repository_inference`.**

A COLPALI adaptation would need:

- parser-derived document units and fallbacks instead of natural-image object
  proposals;
- OCR text, unit type, hierarchy, reading order, and relation features;
- high-resolution handling for tiny document text;
- full-question conditioning where target contents need not repeat question
  wording;
- complete-evidence or carefully qualified weak supervision;
- no-evidence pages and calibrated abstention;
- structural closure for headers, captions, legends, headings, and footnotes;
- fixed-answerer sufficiency and necessity evaluation; and
- measured selector and answerer cost.

## Advantages relative to the current subsystem

**Evidence level: `repository_inference`.**

Relative to the planned direct semantic-unit scorer, a VGent-style decoder
could model candidate complementarity, redundancy, and global set state. It
could also let us change proposal sources without changing the semantic
selection interface.

The direct scorer remains the correct first architecture: it is cheaper,
easier to debug, aligned with the current experiment plan, and able to reveal
whether simpler lexical, visual, and geometric signals already solve the
available supervision. VGent does not justify replacing that baseline before
it is tested.

## Concerns and failure risks

- Natural-image objects and document evidence have different semantics.
- Candidate recall remains a hard ceiling.
- More proposals increase compute and false-positive pressure.
- Frozen VLM hidden states may encode object identity more strongly than
  evidence sufficiency.
- Binary membership does not distinguish partial, complete, complementary,
  and redundant evidence.
- Count targets may impose the wrong evidence decomposition.
- A full VLM pass may increase total cost even when it improves localization;
  cost and evidence quality must therefore be reported as separate outcomes.
- An early semantic hypothesis may select only confirmatory evidence.

## Open questions

- Which VLM hidden states are question-conditioned enough to help without
  requiring answer generation?
- Should global queries estimate candidate count, evidence components,
  question-obligation coverage, uncertainty, or sufficiency?
- Can candidate self-attention improve strict distributed recall at matched
  compute?
- Does a frozen VLM help implicit evidence while merely duplicating OCR
  lexical matching on direct questions?
- How should no-evidence pages interact with global queries?
- Does the semantic gain justify a second multimodal page encoding for the
  intended accuracy-first or efficiency-first deployment setting?

## Smallest decisive experiment

**Evidence level: `unvalidated_proposal`.**

Hypothesis:

> With the same candidates and training rows, a small VGent-style set decoder
> improves strict distributed evidence recall over the direct semantic-unit
> scorer without violating fixed no-evidence and cost constraints.

Control:

- the canonical direct semantic-unit scorer.

Treatment:

- the same candidate embeddings projected into a small decoder with candidate
  self-attention and cross-attention to frozen question-conditioned VLM
  hidden states.

Freeze:

- candidate revision and cap;
- train/dev groups and positive mappings;
- question text and semantic closure;
- threshold-fitting data; and
- downstream answerer and evaluation prompt.

Measure:

- strict distributed R@5 on strong-label rows;
- no-evidence false-positive rate;
- candidate-oracle ceiling;
- fixed-answerer sufficiency;
- selector latency and memory; and
- actual downstream tokens.

Failure interpretation:

- retain the direct scorer if gains disappear at matched compute, occur only
  on answer-string rows, or harm no-evidence calibration.

This experiment is not scheduled or authorized by the active
Evidence-DINO-Units Stage 0–4 task.

## Analysis and interpretation log

- 2026-07-28 — Initial interpretation: VGent is a promising set-decoder
  analogue, not a demonstrated document method. Its best near-term value is to
  define a later matched-candidate ablation if the direct scorer fails on
  distributed or implicit evidence.
- 2026-07-28 — Registry-format decision: retain links and a short factual
  capsule, then use this file primarily for our evolving analysis rather than
  reproducing the paper.

### 2026-07-28 — VLM-semantic proposal selection hypothesis

Paper:
[VGent: Visual Grounding via Modular Design for Disentangling Reasoning and Prediction](https://arxiv.org/html/2512.11099)

**Original VGent mechanism — `paper_reported`:**

VGent decouples semantic understanding from spatial proposal generation:

1. An off-the-shelf detector generates a pool of candidate boxes.
2. A frozen multimodal large language model processes the image and prompt.
3. Hidden states from the MLLM's layers are retained as semantic memory.
4. Each proposal box is mapped through an MLP into a proposal-query
   embedding.
5. A transformer proposal decoder cross-attends from proposal queries to the
   selected MLLM hidden states.
6. Self-attention among proposal queries allows proposals to exchange
   information and be selected as a coordinated set.
7. A binary head predicts whether each detector proposal belongs to the
   requested target set.

This resembles an MDETR-style separation of semantic conditioning and spatial
grounding, but “MDETR for a VLM” is only an architectural shorthand. MDETR
jointly encodes image and text and directly decodes grounded boxes. VGent
instead uses a pretrained MLLM as semantic memory and selects boxes supplied
by external detectors.

**Our project interpretation — `repository_inference`:**

Our current Grounding-DINO experiment is intentionally asking a phrase- and
object-grounding architecture to perform something substantially out of its
original distribution: use a natural document question to identify regions
that may contain the information needed to answer it. The relevant region
often contains text or relationships that are not repeated in the question,
and several regions may be jointly necessary.

A VGent-inspired extension would preserve our high-recall candidate generator
but move the final semantic decision into a document-capable VLM:

```text
document question + page
  -> pretrained document VLM hidden-state memory
  -> high-recall document candidates with fixed geometry
  -> proposal/unit query embeddings
  -> cross-attention to VLM memory
  -> self-attention across candidates
  -> multi-label evidence selection
```

The motivation is to use the VLM's pretrained semantic, OCR, layout, and
reasoning capabilities rather than require MDETR or Grounding DINO to learn
document-question semantics almost entirely from fine-tuning. Candidate
geometry would still come from Grounding DINO, parser-derived units, or their
union, so the VLM would not need to generate coordinates.

This is promising because it preserves the strength of the current plan—a
high-recall, inspectable candidate universe—while potentially improving the
part most likely to be out of distribution: deciding which regions are useful
for answering a natural question. It does not solve proposal recall, complete
evidence supervision, sufficiency, or recovery by itself. Those remain
separate requirements.

**Current status — `unvalidated_proposal`:**

This is a future architecture hypothesis only. The current direct
Evidence-DINO-Units scorer should still be tested first. A VGent-style decoder
becomes justified only if the simpler model has strong candidate coverage but
systematically fails on implicit, distributed, or relational evidence.

### 2026-08-04 — Mid-level document-VGent design with a 2B semantic reranker

**Status — `unvalidated_proposal`:** This section records a candidate
architecture and its scientific motivation. It does not replace the canonical
Evidence-DINO-Units architecture, authorize model acquisition or training, or
change the active Stage 0–4 boundary.

#### The problem this treatment is meant to test

A fixed DeepSeek-OCR unit universe gives us paragraphs, figures, captions,
tables, and fallback regions with stable geometry and inspectable text. That
solves the coordinate-generation problem, but it does not by itself solve the
semantic selection problem. A document question can refer implicitly to a
fact, require a relationship between two units, or need a caption and figure
together even when neither independently appears strongly relevant.

This is precisely where a natural-image phrase grounder such as Grounding DINO
is an incomplete fit. It is useful as a high-recall proposal generator, but
token-to-region matching can behave like an OR over salient query terms. The
answer-bearing document unit may contain none of those terms, and arbitrary
detector boxes do not carry the OCR, type, hierarchy, or figure-caption links
already available from the document parser.

The document-VGent hypothesis is therefore:

> Keep document geometry and candidate construction deterministic, but move
> the final question-conditioned selection decision into a small semantic
> model that can jointly compare all candidates against a pretrained VLM
> representation of the page and question.

A 2B reranker is attractive as the first semantic treatment because it is
small enough to make a controlled screen plausible, already supports direct
relevance scoring without free-form answer generation, and gives us a much
cheaper test than copying VGent's Qwen2.5-VL-7B encoder plus large
LLM-initialized decoder. The 2B size is an experimental cost choice, not a
claim that 2B is known to be sufficient for document evidence reasoning.

The scientific case for this separation is not limited to saving answerer
compute. Evidence localization and answer generation are different outputs.
A semantic localizer may be worthwhile even when total inference is more
expensive if it improves complete evidence recall, provenance, or robustness
under distribution shift. Cost remains a required reported metric, not a
prerequisite that defines whether the architecture is conceptually valid.

#### What transfers from VGent, including in single-target settings

VGent's evaluated task is natural-image object grounding. Its strongest
efficiency and stability motivation concerns multiple target objects, where
autoregressive coordinate generation can stop early, duplicate objects, or
scale poorly with target count. Those specific advantages weaken when a page
contains only one supporting region.

The core semantic-memory/proposal-selection separation nevertheless still
applies to a single target. An ordinary MDETR-style model must learn the
grounding semantics inside a detector architecture and predicts its own boxes.
VGent instead preserves an already pretrained MLLM as the semantic encoder,
accepts geometry from an external high-recall proposer, and performs parallel
proposal membership classification. For documents, the potential gain is not
primarily faster enumeration of many boxes; it is transferring pretrained
document/question semantics into the selection decision without asking the
language decoder to generate coordinates.

Two separate layer mechanisms must remain distinct:

1. **Construction-time initialization:** corresponding LLM-layer weights are
   copied to initialize proposal-decoder layers. This happens when the model
   is built; the copied decoder weights are then their own trainable
   parameters.
2. **Runtime semantic memory:** during each forward pass, intermediate hidden
   states from the frozen MLLM encoder provide the keys and values read by the
   corresponding proposal-decoder layers. These activations change with the
   current image and prompt.

Using several layerwise memories can expose earlier spatial/visual signals as
well as later semantic signals. It does not guarantee that tiny document
details survive the VLM encoder, so local-resolution input remains an
experimental variable rather than a solved issue.

#### Two architectures that must not be conflated

The stock
[Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B)
is the cleanest first VLM baseline. It consumes a query-document pair and
turns the final hidden representation into a scalar yes/no relevance score.
It therefore avoids autoregressive answer or coordinate decoding and output
parsing. It does **not** avoid the 2B multimodal forward pass, and scoring every
unit as a separate cropped document may repeatedly encode overlapping page
content. Its unit decisions are also independent unless we add an explicit
listwise stage.

The VGent-inspired treatment is a custom architecture rather than the stock
reranker interface. It would encode the full page and question once, expose
selected VLM-layer hidden states as shared semantic memory, and let a compact
set decoder score all DeepSeek units together. This could amortize page
encoding and model complementarity or redundancy, but it requires model
surgery, careful layer selection, and new training code. The stock pairwise
reranker should therefore remain a named control rather than be silently
replaced by the custom design.

#### Proposed data and tensor flow

Two input organizations remain open and should be compared rather than
silently conflated:

1. **Full-page memory:** encode the question and full page once. Every fixed
   unit query carries its box and unit features and reads the shared page
   memory. This preserves global layout and amortizes page encoding, but tiny
   unit text may be compressed.
2. **All-unit multi-image memory:** pass every DeepSeek unit crop for the page
   as a separate image in one multimodal input, together with the question and
   unit delimiters. This preserves local resolution and still lets the VLM see
   all units in one forward pass, but it can be token-heavy and depends on the
   VLM's multi-image ordering and causal-attention behavior.

The common selection stage is:

```text
question + (full page or all unit images in one input)
  -> frozen 2B document-capable VLM
  -> selected layer hidden states = shared semantic memory

DeepSeek-OCR semantic/fallback units
  -> ROI visual feature
  -> OCR/text feature
  -> geometry + type + reading-order + generator embeddings
  -> hierarchy and structural-relation features
  -> one initial query embedding per fixed unit

unit queries + shared semantic memory
  -> compact 2–6-layer set decoder
       cross-attention: each unit reads page/question memory
       self-attention: units exchange complementarity/redundancy signals
       relation bias: figure-caption, parent-child, table-header, adjacency
       feed-forward update
  -> scalar relevance per unit
  -> calibrated selection + deterministic structural closure
```

For unit `j`, a suitable initial representation is:

```text
v_j = ROI-pooled visual features for the fixed unit box
o_j = encoded OCR text for the unit
g_j = geometry, type, reading-order, and proposal-source embeddings
u_j = MLP([v_j; o_j; g_j])
```

This deliberately differs from VGent's coordinate-only proposal
initialization. In the paper, a box query initially knows only its coordinates;
question-conditioned content arrives later through cross-attention to the
MLLM memory. Our semantic units already contain legitimate inference-time
information, so discarding OCR, visual content, unit type, and document
structure would waste the main advantage of using parser-derived proposals.

Pairwise relationships should be represented as attention biases or typed
edge embeddings. Arbitrary paragraph IDs, figure IDs, or page-specific hashes
should not be treated as numeric features. Only information available at
inference may enter the model: OCR text, normalized geometry, unit type,
reading order, hierarchy, visual features, generator identity/confidence, and
structural links. Dataset source, answer overlap, gold overlap, mapping
confidence, annotation decisions, and review status remain label or audit
metadata. Generator identity/confidence should be ablated because it may
become a shortcut.

Unlike literal VGent, this version should not regress boxes: the DeepSeek unit
boxes are immutable spatial references. It should also not copy every 2B VLM
layer into an equally large decoder. A compact decoder is the treatment of
interest so the experiment isolates whether VLM semantic memory and joint unit
selection help. A larger localizer may still be defensible for improved
evidence quality, but it would answer a different cost/performance question.

VGent's learnable global queries are not box proposals. In the paper they are
free scratchpad tokens supervised to predict total target-object count and
positive-proposal count. Those targets are coherent for objects but poorly
defined for document units: one semantic support can be represented by a
table, several rows, or cells plus headers.

The default first mini-VGent should therefore contain **no original count
queries and no count loss**. Proposal self-attention already provides joint
candidate interaction. A later ablation may add one or a few global tokens
only if they receive a stable routing target, but that is separate from the
first architecture and must not imply “select exactly N units.”

#### Why not attach a head directly to every VLM unit state?

A direct per-unit head is a useful cheap baseline, but not the main mini-VGent
design. In a causal multimodal sequence, the hidden state assigned to one unit
does not necessarily receive symmetric access to the later question tokens,
every other unit, and the final global interpretation. Input ordering can then
change which unit states know what.

Adding a bidirectional layer across all unit representations corrects that
asymmetry, but it recreates the essential VGent pattern: unit/proposal queries,
self-attention among them, and cross-attention into shared VLM memory. The
matched experiment should preserve the direct-head model as a control and use
the decoder only to test whether symmetric joint selection adds value.

#### Natural-image components excluded from the first mini-VGent

The first model should deliberately omit QuadThinker, GRPO count training,
the two families of object-count queries, SAM mask-aware proposal labeling,
Hungarian box matching, and box regression. DeepSeek already supplies fixed
semantic units, and authoritative evidence can directly supervise multi-label
unit relevance. A figure and its caption can both be positive units in the
same evidence set. Mask-based proposal-label workarounds are unnecessary
unless a later experiment introduces arbitrary overlapping detector boxes.

#### Labels and selection semantics

The document setting does not need VGent's Hungarian box matching or
mask-aware IoA label. We control the candidate universe and should map
authoritative evidence annotations directly onto the smallest available
semantic units:

- **hard positive:** a minimal unit directly covered by authoritative evidence;
- **contextual positive:** a linked caption, header, legend, or similar unit
  needed for interpretation or answer sufficiency;
- **ignore:** an ambiguous overlap or redundant ancestor for which a negative
  claim would be unsafe; and
- **negative:** a defensible distractor from the same page or document.

A figure and its caption can therefore be two positives in one evidence set.
The ground truth remains the authoritative evidence region; DeepSeek units are
the candidate decomposition used to express it. Deterministic structural
closure may add linked context after scoring, but closure must be reported
separately from direct model selection so it does not conceal weak ranking.

Relevance, completeness/sufficiency, and compactness are different targets.
A candidate-level `no` means “prune this unit,” not “skip answering the page.”
Skipping the downstream answerer is safe only when every candidate is below a
calibrated threshold, a separate page-level no-evidence signal agrees, and the
fallback policy has been exhausted. A false-negative early exit is much more
damaging than retaining one extra candidate.

#### Why test this over the main alternatives

| Alternative | What it gives us | Why document-VGent might beat it | Why the alternative may still win |
|---|---|---|---|
| Direct semantic-unit scorer | Cheapest, most interpretable canonical control | VLM memory may improve implicit questions; self-attention can model distributed and redundant evidence | It may already solve the task with far less compute and fewer failure modes |
| Stock 2B pairwise reranker | Fastest way to test pretrained VLM relevance; direct score with no generation | Shared page memory avoids repeated page encoding and a set decoder coordinates units | Stock inference is simpler and better validated; pairwise scoring may be sufficient |
| Grounding DINO over the question | Strong natural-image proposal generation and spatial prior | Fixed semantic units preserve OCR/layout metadata; VLM semantics need not rely on phrase overlap | It may remain the better high-recall proposal source, especially for parser misses |
| Autoregressive VLM emitting boxes or unit IDs | Maximum flexibility and natural-language reasoning trace | Fixed candidates eliminate invalid coordinates, partial enumeration, and generation latency | Generation may express richer plans if calibrated selection proves too restrictive |
| Literal VGent-scale 7B encoder and copied decoder | Closest reproduction of the paper | A 2B encoder plus small decoder is a cleaner and cheaper document hypothesis | Larger models may be necessary for difficult document reasoning |

This architecture is justified only if candidate oracle recall is already high
and the direct scorer fails specifically on implicit, relational, or
multi-unit evidence. If candidate coverage is poor, improving the selector is
the wrong intervention.

#### Main disadvantages and scientific risks

- **Unproven semantic capacity:** the 2B model may not outperform lexical or
  direct multimodal unit scoring on document questions.
- **Total-cost increase:** a high-resolution 2B page forward may consume more
  time and memory than it saves. That does not invalidate an accuracy-first
  localizer, but it rules out an efficiency claim unless measured end to end.
- **Custom implementation risk:** shared hidden-state memory and a set decoder
  are not the stock reranker's supported inference path.
- **Layer-selection ambiguity:** it is unknown which VLM layers provide useful
  OCR, layout, and question-conditioned evidence representations.
- **Candidate ceiling:** parser/OCR omissions and poorly formed units remain
  unrecoverable unless fallback proposals are included.
- **Incomplete supervision:** answer-bearing annotations may omit context
  required for genuine evidence sufficiency.
- **Shortcut learning:** unit type, generator identity, document source, or OCR
  artifacts can correlate spuriously with positives.
- **Candidate-count scaling:** self-attention grows quadratically with the
  number of units unless candidates are capped or staged.
- **Calibration risk:** an overconfident global `no` can erase all downstream
  recovery; early exit must be evaluated separately.
- **Attribution complexity:** simultaneous changes to VLM semantics, shared
  encoding, and set interaction can make gains difficult to diagnose.

#### Clean first experiment

Freeze candidates, positive mappings, splits, negative bundles, metadata,
selection cap, threshold-fitting data, answerer, and answer prompt. Compare:

1. **B0 — cheap control:** lexical/frozen-text unit scorer;
2. **B1 — canonical control:** direct semantic-unit scorer;
3. **B2 — VLM control:** stock frozen 2B pairwise relevance scorer or an
   equivalent independent shallow unit head, with no candidate interaction;
4. **B3 — document-VGent treatment:** the same frozen 2B semantic memory plus
   the compact set-aware unit decoder.

B2 versus B1 isolates the value of pretrained VLM semantics. B3 versus B2
isolates set interaction and shared-page memory. Report candidate-oracle
coverage, strict distributed R@5, R@1, no-evidence false-positive and
false-negative rates, fixed-answerer sufficiency, latency, peak memory,
downstream tokens, and performance by direct versus implicit question type.

For a first architecture screen, use roughly 10,000 balanced training examples
with a fixed development/test set, then confirm a promising result around
25,000 training examples across three seeds. These are experiment-design
targets, not paper-derived numbers. A useful provisional gate is at least
`+5` absolute R@1 over the strongest cheaper control, positive gains across all
three seeds, approximately 90% strong-label R@5, and no unacceptable loss of
no-evidence calibration or end-to-end cost. The exact gate should be fixed
before running the treatment.

## Adoption status and decision history

- 2026-07-28 — `promising`: primary paper checked; no document experiment
  implemented or authorized.
- 2026-08-04 — `promising`: added a nonbinding 2B document-VGent treatment,
  explicit controls, data flow, label semantics, alternatives, and failure
  risks; the direct semantic-unit scorer remains the canonical first model.
