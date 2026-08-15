# MiniVGent Answer-Anchor POC with a HierDoc Region-Policy Baseline

## Proposal metadata

- Status: `unvalidated_proposal`
- Design approval: owner approved continued design on 2026-08-14
- Execution authorization: none
- Active-task effect: none
- Intended dataset: immutable overlap-first document corpus V1
- Intended candidate source: the candidate revision frozen by the approved V1
  segment-reranker baseline
- Primary treatment: compact MiniVGent set decoder over frozen
  Qwen3-VL-Reranker-2B page memory
- Primary structured system baseline: a V1 adaptation of the region policy from
  [HierDoc](https://arxiv.org/abs/2607.29638v1)

This proposal does not authorize model acquisition, candidate regeneration,
training, benchmark execution, synthetic data generation, or changes to the
approved segment-reranker baseline. The current reranker baseline must run
first under its own workspace and SOL handoff.

## Decision

Retain the central MiniVGent architecture:

```text
frozen question-conditioned Qwen page memory
  -> content-rich fixed candidate queries
  -> candidate-to-memory cross-attention
  -> bidirectional candidate self-attention
  -> independent candidate logits
```

Replace the previously proposed global layout-element decoder with the actual
HierDoc comparison. HierDoc's region selector is a generative VLM policy that:

1. sees the question and selected page images with numeric region aliases
   drawn over the candidates;
2. receives a list in the form `alias | type | OCR/table hint`;
3. emits exactly one structured set action such as
   `<evidence_region>3,17</evidence_region>`; and
4. is optimized with a deterministic region-set reward under GRPO.

V1 already supplies the page containing the answer anchor, so the POC bypasses
HierDoc's page policy and tests only its region-selection stage. This is a
faithful adaptation of the region action, not a reproduction of the complete
page-to-region-to-answer system.

The comparison sequence is:

```text
pairwise Qwen ranking
  -> HierDoc region-ID policy on the supplied page
  -> matched small-Qwen autoregressive ID control
  -> shared-page MiniVGent without candidate interaction
  -> full shared-page MiniVGent
```

The HierDoc arm is the external system-level baseline. The small-Qwen
autoregressive ID arm is required to separate HierDoc's useful action design
from its larger backbone, generative decoding, prompt representation, and
GRPO training. MiniVGent versus HierDoc alone is not an architecture-isolated
comparison.

## Source and interpretation boundary

### VGent paper fact

[VGent](https://arxiv.org/abs/2512.11099) freezes an MLLM encoder, maps
externally generated proposals into proposal queries, cross-attends those
queries to layerwise MLLM hidden states, self-attends across proposals, and
uses proposal-level binary classification. It evaluates natural-image
grounding, not document answer-anchor or complete-evidence localization.

### HierDoc paper fact

[HierDoc](https://arxiv.org/abs/2607.29638v1) is a long-document VQA pipeline
with separately optimized page and region policies. The page policy selects a
set of pages. MinerU2.5-Pro parses selected pages into typed regions with
boxes and OCR or table text. The region policy selects a set of sample-local
region aliases. A separate answer model receives the selected full pages,
selected crops, and selected OCR/table text.

Both selectors are initialized from Qwen3-VL-8B-Thinking and trained
independently with GRPO. The paper's region reward weights recall, F1,
precision, and strict format compliance by `0.20/0.50/0.20/0.10`. It does not
train the routing policies from answer feedback. It reports the full
page-to-region pipeline on multi-page and long-document QA benchmarks.

### Project adaptation

The V1 arm transfers HierDoc's actual region-selection interface:

```text
question
  + supplied page with candidate aliases
  + candidate alias/type/OCR list
  -> autoregressive evidence-region ID set
  -> deterministic structured-set reward
```

It changes four things and must report them explicitly:

- the page-selection stage is bypassed because the V1 record already names
  one page;
- frozen V1 candidates replace MinerU2.5-Pro regions;
- accepted answer-anchor alternatives replace complete gold evidence sets;
  and
- evaluation ends at answer-anchor localization unless a separately frozen
  answerer evaluation is authorized.

These changes mean that a successful POC supports only this claim:

> A question-conditioned region-ID policy can select answer-bearing anchors
> from a supplied page and frozen candidate universe.

It does not establish full HierDoc reproduction, multi-page routing,
complete-evidence selection, or document-level answer sufficiency.

## Dataset and label boundary

The POC inherits the immutable V1 corpus and derived artifacts from the
approved overlap-first segment-reranker baseline:

- V1 source hashes and document-grouped splits;
- one frozen `candidate_revision`;
- candidate geometry, member provenance, OCR text, type, and reading order;
- eligible and excluded question manifests;
- accepted answer-location alternatives;
- candidate-oracle report;
- training-only hard-non-anchor and ordinary-non-anchor manifests; and
- byte-identical held-out evaluation inputs.

The labels mean only:

```text
positive: candidate covers an accepted answer-bearing anchor
negative: candidate is audited as not covering an accepted answer-bearing anchor
```

They do not establish complete evidence, necessity, sufficiency, genuine
multi-hop support, or no-evidence behavior. Partial anchors and plausible
unverified context are masked rather than forced negative.

The POC uses no synthetic multi-hop, answer-feedback reinforcement learning,
split redesign, conflict adjudication, or InfographicsVQA model selection.
HierDoc-style GRPO, if later authorized, is a label-based structured-set
objective rather than answer-feedback RL.

## Candidate and action contract

The current DeepSeek semantic segmentation contract does not provide a trusted
nested heading hierarchy or stable parent identifier. Neither HierDoc nor
MiniVGent may assume gold hierarchy depth or parent-child edges.

The first experiment may use only frozen inference-time fields:

- member-box geometry;
- OCR or table text;
- rendered candidate content or ROI features;
- candidate type;
- reading order; and
- typed structural relations that actually exist in the frozen candidate
  manifest.

No dataset source, answer string, gold overlap, review status, conflict flag,
or mapping confidence may enter the model.

For each question-page candidate set, assign consecutive aliases in a
deterministic reading-order traversal. The alias map is a serialization view,
not a new candidate revision. The HierDoc prompt contains:

```text
question
clean page image with visible candidate aliases overlaid
candidate rows: alias | type | OCR/table hint
```

The page image remains authoritative; OCR is a hint. The policy must return
exactly one `<evidence_region>...</evidence_region>` tag containing only listed
numeric aliases. Store the alias map, rendered overlay hash, exact prompt,
decoded text, strict parse result, valid selected set, invalid aliases, and
whether bounded reflection ran.

Do not use gold-derived colors, box styles, ordering, omissions, or prompt
metadata. Alias-rendering collisions and unreadable labels must be measured
before training.

## Experiment arms

### R0 - Stock pairwise Qwen

The released Qwen3-VL-Reranker-2B checkpoint scores each
`(original question, rendered candidate)` independently. This is inherited
from the approved segment-reranker experiment.

### R1 - Tuned pairwise Qwen

The same checkpoint after training-only hard-non-anchor fine-tuning under the
approved reranker protocol. R0 and R1 remain the simplest deployment controls.

If the active baseline selects Jina instead, stock and tuned Qwen must still be
run on the identical frozen manifest before the MiniVGent comparison. A Jina
win does not silently eliminate the Qwen control required by the proposed
MiniVGent backbone.

### H0 - HierDoc-Region zero-shot

Run the paper's region-selection prompt and strict XML action contract using
the frozen Qwen3-VL-8B-Thinking initialization named by HierDoc, subject to an
exact checkpoint-revision freeze. Present the supplied V1 page and V1
candidates instead of pages selected by HierDoc and MinerU candidates.

H0 measures the pretrained policy's ability to follow the action grammar and
localize answer anchors without V1 training. It is not expected to reproduce
paper results because the parser, data, and upstream page stage differ.

### H1 - HierDoc-Region GRPO

Starting from the same frozen revision as H0, optimize the region-ID policy
with group-relative policy updates and the adapted answer-anchor reward below.
Use the paper's reported region settings as the first registered
configuration, not as assumed optima for V1:

- four rollouts per prompt;
- learning rate `1e-6`;
- KL coefficient `0.01`;
- maximum image long edge `1024`;
- region reward weights `0.20/0.50/0.20/0.10`; and
- bounded reflection threshold `8`.

H1 is allowed only if the reward-completeness audit shows that the presented
candidate set can be scored without treating unresolved candidates as
negative. Otherwise retain H0 and defer H1 until supervision is repaired.

### A1 - Matched small-Qwen autoregressive ID control

Use a separately frozen, generative Qwen-family checkpoint at approximately
the same backbone scale as MiniVGent. Give it the identical V1 page, overlay,
candidate list, aliases, and target serialization as H0/H1. Train and evaluate
it on the same rows and update budget as closely as the different objectives
permit.

A1 is not HierDoc. It isolates whether autoregressive candidate-ID generation
itself explains the result. The exact checkpoint and whether it uses SFT,
GRPO, or both must be frozen before execution; no compatible checkpoint is
assumed by this proposal.

### M0 - Shared-page MiniVGent without candidate interaction

Use the proposed full-page Qwen memory and candidate-to-memory cross-attention,
but prevent off-diagonal candidate communication. Preserve the same candidate
features, width, depth, output head, training rows, and loss as M1.

This is an operational control. A diagonal self-attention mask leaves some
self-attention parameters weakly identified, so the report must not call it a
perfect capacity-matched causal control.

### M1 - Full MiniVGent

Use the compact document-candidate decoder:

- complete Qwen3-VL-Reranker-2B backbone frozen;
- question before full-page image in the Qwen sequence;
- selected Qwen language-memory taps after layers 7, 14, 21, and 28;
- candidate OCR, visual ROI, geometry, type, and reading-order features;
- candidate-to-memory cross-attention;
- bidirectional candidate self-attention with only verified relative
  document-structure bias;
- SwiGLU feed-forward blocks; and
- independent candidate logits.

Do not include a count head, cardinality target, box regression, Qwen backbone
LoRA, answer generation, or answer-feedback reward.

## Architecture and representation boundary

HierDoc does not define a global candidate-embedding tensor decoder. Its region
IDs are text tokens generated by a multimodal language model from a page image
and candidate list. It therefore consumes the model's ordinary visual and text
token sequence and produces an autoregressive XML action.

MiniVGent remains the explicit tensorized candidate architecture. For batch
size `B`, maximum candidate count `C`, page-memory length `T`, decoder width
`d=1024`, and selected Qwen memory width `D=2048`:

```text
Qwen layer memories:       H_l in R[B, T, D], l in {7,14,21,28}
candidate visual features: V   in R[B, C, d_v]
candidate OCR features:    O   in R[B, C, d_o]
candidate metadata:        G   in R[B, C, d_g]
candidate mask:            M_c in {0,1}[B, C]
memory mask:               M_t in {0,1}[B, T]

U_0 = MLP([V; O; G])       in R[B, C, 1024]
K_l = W_k,l(H_l)           in R[B, T, 1024]
V_l = W_v,l(H_l)           in R[B, T, 1024]

for decoder block l:
  U <- U + CrossAttention(Q=LN(U), K=K_l, V=V_l, mask=M_t)
  U <- U + SelfAttention(LN(U), mask=M_c, relative_structure_bias=B_rel)
  U <- U + SwiGLU(LN(U))

candidate logits = Linear(LN(U)) in R[B, C]
```

Padded candidates are masked in attention and loss. The output head is
candidate-wise, but candidate states are jointly conditioned in M1. The four
memory taps and their order are source-verified against the proposed Qwen and
Transformers revisions in the
[implementation-readiness companion](minivgent_qwen_implementation_readiness.md).
They still require the real-model parity, shape, visual-grid, and memory
preflight before data training.

The four-block decoder core is approximately 75.6M parameters. Under the exact
candidate encoder and output contract in the implementation-readiness record,
the complete two-block smoke has 42,251,713 trainable parameters and the
complete four-block treatment has 80,074,881. Launch with:

1. a two-block engineering smoke test;
2. a one-seed architecture screen comparing M0 and two-block M1;
3. four-block M1 only if the two-block model is reliable and validation
   supports additional depth; and
4. no six-block or 2,048-wide copied decoder before the compact treatment
   demonstrates value.

H0/H1 and M0/M1 are not parameter- or compute-matched. Report actual trainable
parameters, total parameters loaded, visual tokens, generated tokens, FLOPs
where measurable, peak memory, and latency. A1 supplies the closer
action-space/backbone-scale diagnostic.

## Training objectives

### MiniVGent parallel objective

Independent sigmoid outputs do not by themselves implement accepted
alternative-location semantics. Train M0/M1 with:

```text
L = L_OR_bag + lambda_rank * L_pair_rank
    + lambda_neg * L_verified_non_anchor
```

- `L_OR_bag` requires at least one candidate covering one accepted
  answer-location alternative to score highly.
- `L_pair_rank` ranks a satisfying candidate above retained negatives within
  the same query group.
- `L_verified_non_anchor` applies only to the frozen training-only hard and
  ordinary non-anchor pool after the existing audit rules.
- `partial_anchor`, `unverified_context`, and unresolved candidates receive
  zero loss.

Do not flatten every overlapping candidate or alternative occurrence into a
mandatory positive set. Do not apply a candidate softmax or target-count loss.
The primary POC output is a ranking. Any binary selection threshold is fitted
on validation and frozen before internal-test evaluation.

### HierDoc structured-set reward

For a generated valid candidate set `S`, let `G_a` be the minimal candidate
set for accepted answer-location alternative `a`. Score the best accepted
alternative rather than requiring every alternative occurrence:

```text
anchor_set_score(S) = max_a [
    0.20 * Recall(S, G_a)
  + 0.50 * F1(S, G_a)
  + 0.20 * Precision(S, G_a)
]

R_region = clip_[0,1](anchor_set_score(S) + 0.10 * Format - invalid_penalty)
```

Use HierDoc's invalid-alias penalty `min(0.50, 0.10 * |U|)`, zero reward for a
missing region tag, a `0.25` cap for non-strict output, and a `0.10` cap when
any alias is invalid. These are paper-derived starting values.

This reward is valid only on rows where each presented candidate has a
defensible answer-anchor label or is excluded from reward-bearing action. The
derived manifest must record that decision without mutating V1. Do not let the
precision term silently convert partial anchors or plausible unverified
context into negatives.

Because `G_a` is an answer-anchor target rather than a complete evidence set,
the reward can teach compact anchor selection but cannot teach all information
needed to answer the question. That limitation applies even if downstream QA
improves.

## Multiphase plan

### Phase 0 - Complete the authorized reranker dependency

Required inherited artifacts:

- frozen candidate and eligible-question manifests;
- candidate-count and candidate-oracle reports;
- stock Qwen predictions, or an exact Qwen companion run if Jina was selected;
- tuned Qwen checkpoint and predictions;
- training-only audited negative manifest; and
- matched validation and internal-test manifests.

Preparation stops if candidate coverage is poor or any inherited manifest
changes after reranker scoring.

### Phase 1 - Freeze the HierDoc serialization view

Create a derived, immutable alias and overlay manifest. Verify:

- deterministic alias assignment under a frozen candidate order;
- alias-to-candidate round-trip identity;
- readable labels at the selected image resolution;
- no label overlap that materially hides page content;
- exact candidate-list parity with the visible overlays;
- strict XML parsing and invalid-alias accounting; and
- no gold or audit metadata in prompts or images.

Measure reward-complete row coverage. H1 cannot proceed if its reward would
need to guess labels for unresolved candidates.

### Phase 2 - Representation and systems preflight

Use 128-512 questions stratified by source and candidate count. Verify:

- parity with the official Qwen reranker score path within dtype tolerance;
- H0 prompt/action parsing and bounded-reflection behavior;
- full-page hidden-state hooks for M0/M1;
- exact visual-token count and merged-grid reconstruction;
- manual candidate-to-ROI overlays;
- MiniVGent candidate permutation equivariance;
- HierDoc alias-renaming invariance under matched overlay/prompt remapping;
- no gradients or optimizer state in the frozen MiniVGent backbone;
- finite gradients in every added trainable module;
- peak memory and latency at measured candidate-count p95; and
- tiny deterministic-batch overfit for trainable arms.

No candidate cap is chosen before measuring p50, p90, p95, and maximum counts.
If truncation is necessary, freeze the rule and report post-truncation oracle
loss before training.

### Phase 3 - Zero-shot and action audit

Run R0 and H0 on validation only. Report H0's strict-format rate, invalid-ID
rate, empty-set rate, selected-set size, answer-anchor hit rate, and
best-alternative set scores. Inspect a fixed error sample before deciding that
GRPO is warranted.

This phase tests the actual HierDoc interaction contract before spending on
training and detects overlay or parser failures that model training should not
be asked to repair.

### Phase 4 - One-seed architecture screen

Using a deterministic, source- and document-balanced sample of up to 10,000
eligible training questions, run:

- R1;
- H1 if the reward-completeness gate passed;
- A1 if a compatible checkpoint and training objective were frozen;
- M0; and
- two-block M1.

All arms use identical page/question records and the same frozen candidate
revision. H0/H1/A1 use the same alias view. Keep validation rows, decision
rules, and evaluation code fixed. Equalize optimizer updates where meaningful,
but report rather than conceal irreducible differences in tokens, rollouts,
and compute.

Interpretation:

- `H1 > H0`: V1 structured-set training helps the HierDoc action policy.
- `M1 > M0`: off-diagonal candidate interaction inside MiniVGent helps.
- `M0 > R1`: one shared page memory helps beyond repeated pairwise scoring.
- `A1 ~= H1`: most HierDoc value transfers to a smaller autoregressive ID
  selector.
- `M1 > A1`: parallel tensorized candidate selection helps beyond a matched
  small autoregressive ID action.
- `H1 > M1`: prefer the stronger system if its compute and deployment cost are
  acceptable; do not infer that GRPO or generation alone caused the gain.

### Phase 5 - Confirmatory run

Freeze architectures, prompts, thresholds, reflection rule, reward, and
hyperparameters on validation. Train H1, A1, M0, and the selected M1 depth on
all eligible training questions for three seeds, omitting any arm that failed
its prior gate. Evaluate internal test exactly once after the decision is
locked.

Registered effects use metrics common to the relevant arms:

```text
delta_hierdoc_training = H1 set-F1 - H0 set-F1
delta_shared_memory = M0 Recall@1 - R1 Recall@1
delta_interaction = M1 Recall@1 - M0 Recall@1
delta_parallel_vs_ar = M1 set-F1 - A1 set-F1
delta_system = M1 set-F1 - H1 set-F1
```

`delta_system` is a deployment-system contrast, not a causal architecture
effect.

Proposed promotion gate for full MiniVGent:

- `delta_interaction` at least +2.0 absolute Recall@1 points;
- positive direction in all three seeds;
- document-clustered bootstrap 95% confidence interval excluding zero;
- gain persists when candidate OCR does not contain the accepted answer;
- no material common-metric regression versus R1, A1, or H1;
- competitive accuracy at a matched average selected-candidate count; and
- no failed permutation, alias-remapping, ROI-alignment, manifest-integrity,
  or frozen-backbone check.

The exact minimum effect remains proposed until the binding experiment
specification is approved.

### Phase 6 - Locked robustness evaluation

After all primary choices are frozen, optionally run InfographicsVQA once as a
separate robustness holdout. It must not change the chosen architecture,
threshold, prompt, reward, or checkpoint.

### Phase 7 - Separate multi-page and multi-evidence authorization

A successful answer-anchor POC can motivate two later experiments, neither of
which is authorized here:

1. a full HierDoc-style page-to-region route on multi-page records with
   independently audited page and region evidence sets; and
2. a same-page multi-evidence pilot with 1,000-2,000 training records and
   200-300 separately held-out records, named evidence roles, and proper-subset
   necessity tests.

Do not begin synthetic dependent multi-hop or answer-feedback RL before that
supervision exists and the answer-anchor experiment has been interpreted.

## Metrics

Report candidate-oracle coverage before model metrics.

Common variable-set metrics:

- query-macro best-alternative answer-anchor precision, recall, and F1;
- answer-anchor hit rate;
- average and percentile selected-candidate count;
- accuracy at a matched average selection count;
- empty, malformed, duplicate-ID, and invalid-ID rates; and
- document-clustered bootstrap confidence intervals.

Ranking-arm metrics:

- Recall@1, Recall@3, and Recall@5;
- MRR and nDCG@5; and
- pairwise positive-over-negative accuracy.

Required slices and cost measures:

- candidate type and candidate-count strata;
- source, OCR quality, and answer-box topology;
- answer-string-present versus answer-string-absent;
- direct selection versus bounded reflection;
- latency, peak memory, VLM forward passes, visual/generated tokens, and
  trainable/loaded parameters;
- candidate-order permutation sensitivity for MiniVGent; and
- alias-renaming sensitivity for HierDoc/A1.

The POC has no valid empty-set, complete-evidence, or multi-hop success metric
because V1 does not provide that supervision.

## Failure interpretation

- Poor candidate oracle: repair and refreeze candidate construction; do not
  blame any selector.
- Poor H0 format or alias accuracy: repair the action representation or choose
  a compatible checkpoint before GRPO.
- Insufficient reward-complete rows: do not run H1; improve the audited derived
  supervision view.
- H1 beats H0 but not R1: structured generation learned the task but is not a
  competitive deployment baseline.
- H1 beats M1 but A1 does not: the result may depend on HierDoc's larger model,
  not the region-ID action alone.
- A1 beats M1: retain the simpler autoregressive ID policy unless MiniVGent has
  a decisive cost or robustness advantage.
- M0 beats R1 but M1 does not beat M0: shared page memory is useful, but
  MiniVGent candidate interaction is not.
- M1 gains only on answer-string-present rows: treat it as an OCR shortcut,
  not stronger visual evidence localization.
- M1 beats M0 on single-hop: candidate interaction helps competition or
  redundancy, but complementarity remains unproven.
- All trained treatments fail to beat R1: retain the tuned pairwise reranker.

## Future repository shape after separate execution approval

Binding design and implementation files would be created only after a new
owner authorization:

```text
docs/specifications/overlap_v1_minivgent_answer_anchor_poc.md
docs/superpowers/plans/YYYY-MM-DD-overlap-v1-minivgent-answer-anchor-poc.md
agent-context/research/architecture_registry/experiments/
  overlap_v1_minivgent_answer_anchor_poc.md
reports/overlap_v1_minivgent_answer_anchor_poc/README.md

src/models/qwen3_vl_memory.py
src/models/hierdoc_region_policy.py
src/models/qwen_ar_id.py
src/models/minivgent.py
src/training/minivgent_losses.py
src/training/hierdoc_region_reward.py

scripts/minivgent/profile_representations.py
scripts/minivgent/build_answer_anchor_manifest.py
scripts/minivgent/build_hierdoc_alias_view.py
scripts/minivgent/train_selector.py
scripts/minivgent/evaluate_selector.py

tests/minivgent/test_qwen_memory.py
tests/minivgent/test_hierdoc_region_policy.py
tests/minivgent/test_hierdoc_region_reward.py
tests/minivgent/test_minivgent_model.py
tests/minivgent/test_minivgent_losses.py
tests/minivgent/test_minivgent_evaluation.py
```

The existing candidate builder, reranker adapter, eligibility manifests,
negative audit, and ranking evaluator should be reused rather than duplicated.

## Adoption boundary

This record is a durable, owner-approved research design and remains an
unvalidated proposal. It does not change `CURRENT_TASK.md`, the active V1
segment-reranker specification, the SOL task, or the experiment workspace.

Moving it into `experiments/` requires a separate explicit authorization that
freezes the HierDoc and Qwen revisions, candidate revision, alias rendering,
data manifests, rewards, metrics, promotion gate, workspace, and SOL handoff.
