# Later HierDoc and Answer-Sufficiency Experiment

## Status

- Scientific direction: retained and owner-designated as a later test.
- Active POC effect: none.
- SOL authority: none.
- Dependency: complete the Qwen R0/R1 versus MiniVGent M0/M1 POC first.
- Supervision dependency: independently audited complete/sufficient evidence
  sets or a validated frozen-answerer sufficiency procedure.

This file preserves the HierDoc and A1 experiment design without allowing it to
confound or expand the initial MiniVGent POC.

## Why it is later

HierDoc predicts compact region-ID sets and trains with recall/F1/precision
reward. Answer-anchor V1 labels do not establish that every other selected
region is unnecessary, so precision-bearing reward can penalize useful
unlabeled context. HierDoc is more informative once the target is answer
sufficiency: select enough evidence to answer correctly and compactly, while
testing whether removal of a selected unit breaks the answer.

The later comparison also needs A1. Without a similarly scaled
autoregressive-ID control, a HierDoc versus MiniVGent result confounds the
region action with the 8B backbone, prompt, decoding, reflection, and GRPO.

## Preserved arms

- **H0:** zero-shot HierDoc-style region-ID policy on the frozen supplied page
  and candidate aliases.
- **H1:** the same policy after structured-set training, only under complete or
  reward-complete evidence supervision.
- **A1:** matched small-Qwen autoregressive-ID control with the identical page,
  aliases, candidate list, and action grammar.
- **M1-S:** the frozen/promoted MiniVGent architecture retrained or evaluated
  only under the later sufficiency targets.
- **R1-S:** tuned pairwise reference under the same later candidate/evidence
  view.

The suffix `-S` prevents later answer-sufficiency results from being confused
with the V1 answer-anchor POC.

## Required supervision before activation

Each later row must establish:

1. the complete evidence set supports the accepted answer under an independent
   verifier or frozen answerer;
2. no proper subset is sufficient, or every selected unit has a documented
   evidence role and leave-one-out effect;
3. plausible unlabeled context is not silently labeled false;
4. alternative sufficient sets are represented with OR semantics;
5. document split assignment precedes any generation or augmentation; and
6. the held-out sufficiency evaluation is independently audited.

A proposed first dataset remains approximately 1,000-2,000 training records
plus 200-300 separately held-out records covering table header+value,
comparison rows, claim+qualifier, chart mark+legend, definition+application,
and answer span+disambiguating heading. These are planning estimates, not
validated counts.

## Candidate and serialization dependency

The experiment may reuse the POC's frozen candidate revision if its oracle is
adequate for complete evidence. Otherwise it must create and audit a new
candidate revision before model comparison. It then freezes:

- deterministic aliases and overlay/list parity;
- one strict `<evidence_region>...</evidence_region>` grammar;
- gold-neutral rendering and alias-renaming tests;
- reflection threshold and accounting;
- common candidate omissions/masks; and
- prompt, processor, page, candidate, and action hashes.

The detailed preserved contracts live in:

- `../../architectures/hierdoc_region_policy.md`;
- `../../architectures/qwen_autoregressive_control.md`; and
- the primary-source HierDoc registry review.

## Later stages

### L0 - Sufficiency dataset and oracle

Build/audit complete or pseudo-minimal evidence sets. Report candidate oracle,
proper-subset failures, verifier agreement, rejected ambiguity, and source
bias. Stop if precision/necessity cannot be grounded.

### L1 - Alias/action preflight and H0

Freeze the HierDoc/A1 serialization, parser, reward, reflection, and model
locks. Run H0 validation-only and audit strict format, invalid IDs, alias
renaming, set size, answer sufficiency, and cost.

### L2 - One-seed H1/A1/M1-S screen

Train H1 and A1 under the exact later target/reward, and compare with M1-S and
R1-S on identical rows. Report answer correctness, evidence groundedness,
complete-set recall/F1, proper-subset necessity, compactness, and cost.

### L3 - Confirmatory answer-sufficiency test

Freeze all choices on validation, run three seeds, open the later held-out set
once, and compute document-clustered uncertainty. A downstream answer model,
if used, remains frozen across selector arms.

### L4 - Possible full-document extension

Only after page-level sufficiency succeeds, separately test HierDoc-style page
routing. Report page oracle/recall before region performance because omitted
pages are irreversible.

## Preserved HierDoc starting configuration

- Qwen3-VL-8B-Thinking initialization named by the paper;
- four rollouts per prompt;
- learning rate `1e-6`;
- KL coefficient `0.01`;
- maximum image long edge `1024`;
- reward weights `0.20/0.50/0.20/0.10` for recall/F1/precision/format;
- invalid penalty `min(0.50, 0.10 * invalid_count)`;
- missing tag reward zero;
- non-strict cap `0.25`;
- any-invalid-alias cap `0.10`; and
- bounded reflection above eight selected regions.

These are paper-derived starting settings, not assumed optima.

## Answer-feedback progression

If native audited sets are insufficient, use a frozen answerer first for
search/distillation rather than online RL:

1. start from the answer anchor;
2. add candidates from a bounded plausible neighborhood;
3. score answer correctness and groundedness;
4. prune by leave-one-out tests;
5. reject ambiguous cases; and
6. distill accepted pseudo-minimal sets with supervised training.

Only after the action grammar and sufficiency verifier are reliable may a new
policy-optimization experiment combine answer correctness, groundedness,
corrupted/empty-evidence improvement, selected token/area cost, redundancy,
and abstention. Freeze the answerer and candidate universe to prevent
co-adaptation shortcuts.

## Non-automatic boundary

Completion of the active POC does not activate L0. Before any later job, create
a dedicated binding specification, implementation plan, workspace record, SOL
handoff, exact model locks, sufficiency dataset/version, reward, metrics, and
promotion gate.
