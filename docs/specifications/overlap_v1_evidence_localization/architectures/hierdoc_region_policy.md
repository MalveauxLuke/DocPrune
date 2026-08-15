# HierDoc Region-Policy Architecture

## Status

This architecture is preserved for the later answer-sufficiency experiment.
It is not an active MiniVGent POC arm, implementation task, model-download
authorization, or SOL stage. Its later activation depends on complete or
independently verified sufficient-evidence supervision and a separate handoff.

## Role and source boundary

H0/H1 later adapt the region-selection stage from HierDoc, arXiv
`2607.29638v1`, as a structured answer-sufficiency comparison:

- **H0:** zero-shot region-ID action under the frozen paper-named
  Qwen3-VL-8B-Thinking initialization;
- **H1:** the same policy after V1 answer-anchor structured-set training, only
  if reward-complete supervision is verified.

No official implementation was identified in the reviewed paper. Paper facts,
V1 adaptations, and local measurements must remain explicitly separated.

## Paper-reported system

HierDoc is a three-stage long-document VQA pipeline:

```text
page policy:   (question, all pages) -> selected page-ID set
region policy: (question, selected pages, parsed regions) -> region-ID set
answer model:  (question, selected full pages, crops, OCR/table text) -> answer
```

The paper:

- divides documents into non-overlapping windows of at most 16 pages;
- unions window-level page selections;
- runs bounded reflection if more than eight pages are selected;
- parses selected pages with MinerU2.5-Pro into typed, boxed candidates with
  OCR or table text;
- gives the region policy alias-overlaid pages and candidate rows;
- requires one XML region-set action;
- uses analogous reflection above eight selected regions; and
- sends both selected full pages and selected local crops/text to a separate
  answer model.

Page and region policies initialize from Qwen3-VL-8B-Thinking and are trained
independently for one epoch with GRPO. The paper maps a gold evidence box to a
parser region when box IoU is at least 0.10 or at least 0.80 of the parser
candidate area lies inside the gold box. Same-document non-evidence pages add
training distractors.

Reported region settings:

| Field | Value |
|---|---:|
| rollouts per prompt | 4 |
| learning rate | `1e-6` |
| KL coefficient | `0.01` |
| maximum image long edge | `1024` |
| reflection trigger | more than 8 selected regions |
| reward recall/F1/precision/format weights | `0.20/0.50/0.20/0.10` |

The invalid-alias penalty is `min(0.50, 0.10 * invalid_count)`. Missing region
tag receives zero reward, a non-strict action is capped at 0.25, and any action
with an invalid alias is capped at 0.10.

The paper's region corpus starts at 2,418 ViDoRe-v3 records and retains 1,022
after parser alignment/audit. Its MMLongBench-Doc incremental ablation reports
5.51% accuracy and 4.82% F1 improvement from the trained/reflected region stage
over selected full pages alone. Removing full pages and retaining only crops
and OCR lowers performance, suggesting complementary global and local views.
These are paper-reported results, not local expectations or measurements.

## V1 adaptation

V1 supplies one answer page per question, so this program bypasses HierDoc's
page-selection stage. Frozen DeepSeek-OCR-2 V1 candidates replace MinerU
regions. Accepted answer-anchor alternatives replace complete gold evidence
sets. Evaluation stops at anchor localization; no answer model is part of the
program.

The supported claim is:

> On one supplied page, a question-conditioned region-ID policy can select an
> accepted answer-bearing anchor from a frozen semantic candidate universe.

Under native V1 anchors alone, this would not be a full HierDoc reproduction,
page-routing experiment, complete-evidence selector, or downstream QA
evaluation. The later experiment must replace that weak target with audited or
independently verified sufficient-evidence sets before H1.

## Input serialization

For every question-page candidate set:

1. sort frozen candidates deterministically by reading order and candidate ID;
2. assign consecutive sample-local numeric aliases;
3. render those aliases on a clean page without gold-dependent styling;
4. provide candidate rows `alias | type | OCR/table hint`; and
5. retain the exact alias, overlay, list, prompt, page, and candidate hashes.

Logical input:

```text
original question
clean supplied page with visible aliases
candidate rows: alias | type | OCR/table hint
```

The page image is authoritative; OCR/table text is a hint. The overlay and list
must contain exactly the same candidate set. Gold fields, source identity,
conflict state, mapping confidence, answer strings, relation labels, and
gold-derived colors/order/omissions are forbidden.

## Action grammar

The policy emits exactly one tag:

```xml
<evidence_region>3,17</evidence_region>
```

The parser retains:

- decoded text;
- strict versus non-strict parse state;
- listed and duplicate aliases;
- invalid aliases;
- valid selected `candidate_id` set;
- empty/missing/malformed action;
- whether bounded reflection ran; and
- first-pass and reflected action hashes.

Do not silently repair, discard, or convert invalid generations. Duplicate IDs
count as a format diagnostic even when set semantics deduplicate them.

## H0

In the later L1 stage, H0 freezes an exact
Qwen3-VL-8B-Thinking-compatible checkpoint revision,
environment, prompt, processor, image settings, decoding configuration,
reflection rule, and parser. It receives the V1 page/candidates rather than
paper MinerU candidates or an upstream page-policy result.

H0 measures zero-shot grammar following and answer-anchor localization. It is
not expected to reproduce paper metrics. Validation-only reporting includes
strict-format, missing/malformed, invalid-ID, duplicate-ID, empty-set,
selection-size, reflection, anchor hit, best-alternative set metrics,
alias-renaming sensitivity, latency, tokens, and memory.

## Reward-completeness gate

HierDoc's precision/F1 reward assumes that non-gold selections are known
false. V1 does not supply complete evidence labels. H1 is legal only on a
derived reward view in which every reward-bearing candidate is defensibly:

- a member of one accepted positive alternative; or
- a verified non-anchor after the program audit.

Partial anchors, unresolved candidates, plausible context, and unverified
non-anchors must be removed from reward-bearing action or the row excluded.
The reward-view manifest records the presented candidate set, masked/excluded
candidates, target alternatives, reason, and hashes without mutating V1.

Report reward-complete row coverage and its source/candidate-count bias. If H1
would need to treat unknown candidates as false, H1 stops while H0 remains
valid.

## Adapted H1 reward

For valid selected set `S` and each accepted minimal anchor alternative `G_a`:

```text
anchor_set_score(S) = max_a [
    0.20 * Recall(S, G_a)
  + 0.50 * F1(S, G_a)
  + 0.20 * Precision(S, G_a)
]

R_region = clip_[0,1](
  anchor_set_score(S)
  + 0.10 * strict_format
  - min(0.50, 0.10 * invalid_alias_count)
)
```

Then apply paper-derived behavior:

- missing region tag: reward 0;
- non-strict action: cap total reward at 0.25;
- any invalid alias: cap total reward at 0.10.

Score the best OR-equivalent accepted alternative, not their union. These
paper-derived values are the first registered configuration, not assumed V1
optima. H1 uses four rollouts, LR `1e-6`, KL `0.01`, long edge 1024, and
reflection trigger 8 unless Stage 03 registers a justified configuration
change before optimization.

The reward teaches compact answer-anchor selection. It cannot teach all useful
information for answering, even if a later answerer happened to improve.

## Reflection

Run at most one bounded second pass when the first pass selects more than eight
valid regions. The reflected prompt may contain only the first-pass selected
candidate subset under a deterministic remapping recorded in the artifact.
Reflection changes latency and generated tokens and is reported separately.

Reflection cannot rescue candidates absent from the original frozen universe,
cannot use gold feedback, and cannot be activated selectively from target
status.

## Required invariance and representation tests

Before later H1 training:

- alias-to-candidate round trip is exact;
- overlays and lists have set parity;
- labels are readable and do not materially hide content;
- strict parser preserves every failure mode;
- deterministic decoding is reproducible where configured;
- remapping aliases in both overlay/list/action evaluation does not materially
  change semantic selection;
- gold-neutral styling/order checks pass;
- bounded reflection has exact first/second-pass accounting; and
- reward tests cover alternative targets, invalid caps, missing tags, masked
  candidates, and reward-incomplete refusal.

## Interpretation and limitations

- `H1 > H0`: V1 label-based structured-set training helps the region policy.
- `H1 <= R1`: the structured action learned or followed the task but is not a
  competitive deployment system.
- `H1 > M1` without `A1 > M1`: the difference may come from the 8B backbone,
  not generation or GRPO alone.
- `H1 > M1` with acceptable cost supports deploying H1 as a system, not a
  causal claim that autoregression or GRPO is superior.
- invalid or alias-sensitive behavior is a representation failure, not a
  localization result.

H0/H1 and M0/M1 are not compute- or parameter-matched. Report trainable and
loaded parameters, VLM passes, input and generated tokens, FLOPs where
measurable, latency, memory, and reflection frequency.

## Failure risks retained from the paper review

- parser/candidate recall is a hard ceiling;
- page misses would be irreversible in a future full-document system;
- aliases can obscure tiny text or create rendering shortcuts;
- autoregressive output can be malformed, incomplete, duplicated, or invalid;
- precision reward is unsafe under incomplete labels;
- the 8B GRPO policy is not compute matched to MiniVGent;
- reflection is a heuristic extra inference pass; and
- local crops should not be assumed to replace global page layout in a future
  answering system.

## Further deferred HierDoc extensions

Page policy training, 16-page window routing, same-document distractor pages,
full pages plus local crops fed to an answer model, and downstream QA are
future separately supervised experiments. They are paper context, not hidden
features of H0/H1.
