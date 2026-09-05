# Chat summary: DocPrune CTP fidelity and random-control experiment

Conversation state summarized through 2026-08-27; archived 2026-09-05.

## Key takeaways

1. The canonical full top-4 benchmark showed a real but modest quality gap:
   all-kept scored `37.7911 F1` and the then-current DocPrune reconstruction
   scored `36.7603 F1`. On the 2,126 questions whose ordered cached pages
   matched exactly, the controlled DocPrune delta was `-1.1980 F1`, with paired
   interval `[-2.3471, -0.0640]`.
2. A 245-question stage diagnostic localized the main controlled loss to CTP:
   `44.8612` all-kept -> `43.1143` BTP-only -> `43.6327` BTP+QTP -> `41.3347`
   full DocPrune. Adding CTP lost `2.2980 F1`, with paired 95% interval
   `[-4.2980, -0.6286]`.
3. BTP and QTP contain reconstruction ambiguity, but targeted alternatives did
   not explain the paper/local discrepancy. Small BTP raster variants barely
   changed masks. A promising QTP head-mean variant failed on the full
   245-question development diagnostic and was rejected.
4. The strongest CTP hypothesis became aggregation of query-to-visual
   **pre-softmax logits across heads, followed by a visual softmax**. It better
   matched the paper-derived retention fingerprint and performed better than
   the literal/current reconstruction, but it still did not reproduce the
   paper's reported CTP quality gain. It must be described as the
   best-supported local reconstruction, not the unpublished official method.
5. The manual decoder's all-kept path was made behaviorally equivalent to
   stock generation, including EOS/preprocessing and forced-boundary behavior.
   This removed an important confound and made the implementation suitable as
   a controlled baseline for comparing token-selection methods.
6. The approved comparison is not generic "random pruning." Random and
   coverage controls operate at the same post-block boundary, over the same
   post-BTP+QTP visual population, and at the exact same per-question budget as
   the corresponding CTP method. Physical deletion, cache compaction, original
   token order, and M-RoPE values are preserved.
7. Task 6 reached a four-GPU smoke. All corrected jobs completed, but the
   preregistered exact cross-GPU mask comparator rejected them because QTP kept
   3,586--3,597 tokens depending on GPU. This was a maximum count spread of 11
   tokens, about `0.31%`, while deterministic answers were identical. The user
   approved a dated amendment allowing internally consistent L40S runs to be
   canonical and treating other GPUs as numerical-drift evidence only.
8. At the end of this chat, the 64-question, 45-cell-per-question L40S matrix
   had **not been confirmed submitted**. A first handoff review found three
   launch-contract blockers; they were fixed test-first in clean successor
   commit `2d2bea2722774e29009269637aa225131f170fd4`. Final independent re-review
   was interrupted or its verdict was not received in the captured context.
9. Never run fresh/global retrieval for these controlled diagnostics. Reuse
   the sealed ordered pages and persisted page features. This rule was added to
   repository and SOL guidance after the user objected to any possibility of
   recomputing retrieval.

## Archival and authority caveat

The active repository during the conversation was
`/home/lmalveau/DocPrune-fix-evaluate-measurement`. On 2026-09-05 that path no
longer existed. This summary was therefore saved in the surviving repository
`/home/lmalveau/DocPrune` at main commit
`d9bb7dda02b8abfb7a570fac0ddf8213ffb1c373`.

The surviving repository also has later Task 9 worktrees, so work evidently
continued after this chat. This summary is historical knowledge, not current
SOL authority. Before acting, read the current `AGENTS.md`,
`agent-context/INDEX.md`, `agent-context/CURRENT_TASK.md`,
`sol/AGENTS.md`, and `sol/CURRENT_SOL_TASK.md`, and reconcile later Task 6--9
records. Do not revive an old handoff merely because it is described here.

## Original requested workflow and durable constraints

The user first asked the agent to read, in order:

1. `docs/reproduction/docprune_handoff_2026-08-24/README.md`;
2. `docs/reproduction/docprune_handoff_2026-08-24/REFERENCE.md`;
3. `sol/CURRENT_SOL_TASK.md`; and
4. all relevant `AGENTS.md` files.

The user asked for a review of recent jobs, an explanation of the discrepancy
with the paper, a codebase-only fidelity audit, and a systematic diagnosis of
BTP and CTP. Important operating instructions were:

- preserve every existing artifact and all uncommitted work;
- use the active lightwork CPU allocation directly for CPU work;
- use many short GPU jobs across GPU sizes for diagnostics;
- keep canonical quality on one GPU family and use other families for drift;
- never recompute retrieval unless explicitly requested;
- use existing cached ordered pages and persisted features;
- do not load the global FAISS index, embedding tensor, or token map for fixed
  controlled runs;
- prefer paper-author code before synthesizing local equivalents;
- record paper omissions, local adaptations, failed runs, hashes, and course
  changes;
- continue autonomously until a real approval/input boundary is reached.

The user repeatedly challenged assertions that "everything in the paper was
copied." That correctly changed the task from defending the implementation to
a codebase-derived audit separating:

- behavior directly supported by paper text or released author code;
- necessary architecture adaptations for Qwen2-VL/DocPrune;
- implementation choices the paper does not specify; and
- hypotheses selected because they match observed fingerprints rather than
  because the paper explicitly defines them.

## Canonical baseline and recent benchmark jobs

### Full top-4 benchmark

The top-4 benchmark ultimately completed for all `2,441/2,441` questions in
both all-kept and then-current DocPrune modes. Work was sharded because earlier
monolithic or packaging jobs were too costly or memory-heavy.

Relevant job families recorded during the chat included:

- initial paired checkpoint arrays `62008122` (DocPrune) and `62008123`
  (all-kept), shards 0--3;
- initial continuation arrays `62030328` and `62030329`;
- recovery arrays `62041375` and `62041383` after untracked M3DocRAG bytecode
  made a clean-check fail before evaluation;
- mixed-hardware quality-only replacement arrays `62068296` (all-kept) and
  `62068302` (DocPrune).

Mixed hardware was allowed only for quality completion in that historical
benchmark. Efficiency aggregation across hardware was forbidden.

The sealed paired-quality analysis was:

`/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/analysis/top4-paired-quality.json`

Its file SHA-256 was
`e07cc815a2847e6a9af21761fc29f0b1241f4e411b54132bb65b04cf47c3b51a`;
its internal digest was
`2770fc16f66cc5f33d856179032335a03f8da8a0abc5e967780eff5dedd6ea14`.

Canonical published quality-only results were:

| Mode | Rows | F1 | Result SHA-256 | Summary SHA-256 | Manifest SHA-256 |
|---|---:|---:|---|---|---|
| All-kept | 2,441 | `37.791069233920446` | `ace39b9e...07b1` | `9e634aa1...649c` | `6ef2c6bb...189` |
| DocPrune | 2,441 | `36.760344121261724` | `fb148d6c...b217` | `5dc26558...c5ac` | `bb54cb34...95b` |

The shortened hashes above were recorded in full in the then-current
`sol/CURRENT_SOL_TASK.md` and experiment log.

### Memory-heavy merger diagnosis and fix

Quality packaging jobs `62086624`/`62086625` with 32 GiB and
`62087740`/`62087864` with 64 GiB failed by host OOM and published nothing.
The cause was not insufficient arbitrary memory: the merger redundantly ran
deep index validation for each already-validated shard, materializing roughly
23.9 GB of embeddings, a 2.8 GB JSON token map, and a 23.5 GB FAISS index.

The lightweight merger fix was sealed at commit
`a53bf95b8e4288f180553f04b4fa1eec8a841df5` in clean runtime
`/home/lmalveau/DocPrune-quality-runtime-a53bf95`. It authenticated saved result
and validation-file hashes, the plan digest, shard classification, and
immutable manifest identity without loading the global index. Two predecessor
CPU invocations failed closed because production provenance fields absent from
tiny tests were not yet forwarded; both published nothing. After production-
shaped regression tests, direct lightwork CPU publication completed
atomically. This satisfied the user's first requested priority.

## Locating the quality loss by stage

An initial 64-question diagnostic used an outcome-blind selection from the
available paired single-hop questions. Several attempts demonstrated why
fail-closed provenance mattered:

- arrays `62063320` and `62063380` were canceled without results because they
  had no estimated start;
- replacements `62065503` and `62065508` failed before evaluation because the
  nominal runtime worktree no longer matched its sealed commit;
- retries `62071272` and `62071274` failed before the first question because a
  non-finite CTP-disable threshold violated controller validation;
- corrected runtime commit `607fc38e...` used the established finite `1e9`
  disable threshold;
- BTP shard recovery `62074548_1` restored exact fixed-page identity after an
  H100 result had drifted on 9/32 page lists and was excluded.

The controlled 64-question F1 trajectory was:

`41.0625` all-kept -> `39.265625` BTP-only -> `39.5625` BTP+QTP -> `39.625`
full DocPrune.

The adjacent paired intervals included zero, so the diagnostic expanded to
the existing 245-question development cohort. Arrays `62075807` (BTP-only)
and `62075816` (BTP+QTP) completed all remaining shards. The 245-question
trajectory was:

`44.8612` all-kept -> `43.1143` BTP-only -> `43.6327` BTP+QTP -> `41.3347`
full DocPrune.

The CTP increment was `-2.2980 F1`, paired 95% interval
`[-4.2980, -0.6286]`. This is the strongest evidence from the chat that the
controlled discrepancy is primarily CTP, not BTP or QTP.

## Paper/code fidelity audit

The audit was explicitly required to use repository code and paper evidence,
not conversational assumptions. Durable audit files in the original worktree
included:

- `docs/reproduction/CODE_PAPER_FIDELITY_AUDIT.md`;
- `docs/reproduction/PAPER_AMBIGUITY_AUDIT_2026-08-26.md`;
- `docs/reproduction/DISCREPANCY_AUDIT.md`;
- `docs/reproduction/RECONSTRUCTION_GAPS.md`; and
- `docs/reproduction/FAIR_CTP_BASELINE_2026-08-26.md`.

The major conclusion was that broad pipeline stages could be paper-faithful
while critical low-level attention semantics remained under-specified. The
paper did not fully determine such matters as:

- which attention tensor is thresholded;
- pre-softmax versus post-softmax values;
- normalization domain;
- aggregation order across query heads;
- treatment of Qwen grouped-query/KV heads;
- exact layer-input versus layer-output comprehension measurement;
- precise generation timing and first-token semantics; and
- some preprocessing/EOS and cache-deletion details.

Therefore an implementation can copy every stated equation and hyperparameter
yet still differ materially from the authors' unpublished code.

## BTP findings

BTP removes background-like visual groups. The audit considered whether the
paper meant mean pixels, all pixels, different grayscale formulas, inclusive
versus strict tolerance, or other raster details.

An independent CPU raster reconstruction over the same 16 diagnostic
questions reproduced every saved current BTP mask across 160,512 visual
groups. BT.601 truncation changed only 49 groups, channel-mean grayscale only
50, and inclusive pixel tolerance 989. Retention moved only from about
`52.2453%` to `51.6292%`; mean mask Jaccard for the inclusive variant was
`0.9880`.

The BTP head/group screens found:

- BTP `mean` changed every tested mask (mean Jaccard `0.5887`) but was neutral
  on the 12 calibration questions;
- BTP `all` retained only `6.93%` and was rejected as implausibly aggressive.

Conclusion: BTP may not be provably identical to unpublished author code, but
ordinary grayscale, rounding, tolerance, and mean/all variants cannot explain
the observed paper/local quality gap. No further GPU time was recommended for
those variants.

## QTP findings

QTP uses query relevance after BTP. The conversation recognized that Qwen2-VL
grouped-query attention creates choices not specified by the paper, including
averaging query heads within each KV group, averaging before or after KV
repetition, selecting heads, or weighting heads.

The user correctly noted that an unknown head rule could make a method brittle.
The response distinguished two points:

- if performance depends on an undocumented choice of heads, reproduction is
  brittle;
- grouped-query aggregation can nevertheless be architecture-grounded rather
  than arbitrary, because several query heads share each KV head.

A QTP `mean` candidate initially improved the small calibration sample:
`35.1667 -> 46.2500 F1`, with 2 wins, 0 losses, 10 ties and a wide interval
`[0.0, 27.75]`. It changed all 16 masks modestly (mean Jaccard `0.95749`) and
reduced combined retention from `45.50%` to `43.64%`. That apparent win did not
survive expansion. On all 245 development questions, current BTP+QTP scored
`43.6327`, while QTP-mean scored `42.6653`, delta `-0.9673 F1`, with 7 wins,
229 ties, 9 losses and interval `[-3.1755, +1.0408]`. QTP-mean was rejected.
QTP `all` had already lost `8.3333 F1` on calibration and was rejected.

Conclusion: retain the current QTP semantics for the controlled baseline.
They remain a reconstruction, not a guaranteed match to unpublished author
code.

## CTP attention-semantics investigation

### Why the original/current CTP looked wrong

The early audit found that the current implementation averaged attention
across heads and then multiplied by the visual-token count before applying the
paper threshold. The paper referred to thresholding "attention weights" but
did not specify that multiplication. This placed the threshold on a different
score scale and was initially the strongest explanation for over-pruning.

Subsequent tests showed that normalization scale alone was not enough. A
candidate could restore the paper-like number of tokens but retain the wrong
token identities and fail to restore answers. The remaining discrepancy was
therefore thought more likely to involve how attention values were obtained
and aggregated.

### Plausible interpretations discussed or tested

The candidate space included:

- direct post-softmax attention thresholds;
- visual-only renormalization;
- mean or sum across heads;
- raw `qk^T/sqrt(d)` logits;
- averaging logits before softmax;
- grouped-query/KV-aware aggregation;
- max-head scoring;
- original-token versus post-QTP scale; and
- attention captured during prefill versus from the first generated token.

The user specifically asked why pre-softmax logits should not be considered.
The answer was that their numerical threshold magnitudes were plausible, but
the paper's wording "attention weights" normally suggests post-softmax values.
Because the paper was ambiguous and no author code was available, it remained
a legitimate diagnostic rather than a faithful-by-text conclusion.

The user also noted that a visual-softmax test had already been performed; the
analysis was adjusted to avoid proposing it as new work. The user approved a
sum-across-heads test and asked whether 16 questions were enough. The response
treated 16 as sufficient for rapid screening and mask/retention fingerprints,
but not for statistically reliable quality conclusions. Promising candidates
were to be expanded to the existing 64 or 245 development cohorts.

### Rejected CTP alternatives

- Literal sum-before-softmax was decisively implausible on authenticated CPU
  captures: at threshold `0.5` it retained only `0.1844%` of top-1 calibration
  tokens (`25/13,555`) and its `0.1--0.7` curve was nearly flat.
- Max-logit/max-head aggregation was far from paper-derived targets. The live
  max-head candidate retained about `82%` of QTP survivors versus the
  paper-derived `65%` target and reduced calibration F1 by `6.6667` points.
- Original-token scaling raised conditional CTP retention from `41.3069%` to
  `58.0067%`, closer to the paper's token quantity, but calibration F1 fell
  `46.8333 -> 43.0833` (`-3.75` points; 0 wins, 1 loss, 11 ties; interval
  `[-11.25, 0]`). It was rejected as a quality fix.

These results support an important distinction: current CTP over-prunes by
count, but merely adding more tokens does not recover the paper result because
the selected identities also matter.

### Aggregate-logit candidate

The most viable hypothesis averaged raw per-head query-to-visual logits before
applying a visual softmax. On authenticated CPU captures, its paper-derived
top-1/top-2/top-4 conditional retentions were approximately:

- top-1: `33.7809%`;
- top-2: `38.8091%`;
- top-4: `67.4313%`;

against targets of roughly `35.8491%`, `41.8182%`, and `65.0%`, for mean
absolute error `2.5029` percentage points.

For top-1 thresholds `0.1`, `0.3`, `0.5`, and `0.7`, its retention curve was
about `71.49%`, `45.67%`, `33.78%`, and `26.96%`. The canonical L40 recovery
array `62169007` completed all 16 cells. Thresholds `0.3`, `0.5`, and `0.7`
produced identical candidate answers. Threshold `0.1` corrected one small-
sample yes/no answer; the other 11 calibration answers and all four stress
answers were unchanged across thresholds. This resembles the paper's
threshold-sensitivity fingerprint: retention/throughput can change while
quality remains nearly flat.

The chat also cited a stage-64 aggregate-logit analysis at:

`/scratch/lmalveau/docprune/ctp-aggregate-logit-stage64-fixed-v1/analysis.json`

It was described as improving approximately `+1.08 F1` over current CTP and as
encouraging but not statistically conclusive. It still performed worse than
BTP+QTP, whereas the paper reported CTP improving performance. Thus it is a
better reconstruction candidate, not a solved reproduction discrepancy.

### Grouped-query-head candidate

A CPU head audit found an architecture-grounded candidate: average the seven
query-head logits sharing each Qwen KV head, softmax within each KV group, then
average the four group maps. It matched the paper-derived 1/2/4-page retention
fingerprint more closely than the aggregate-logit candidate (`~1.72` versus
`~2.50` percentage-point mean error) and changed roughly `9%` of the selected
top-4 mask.

This was added or proposed as an explicitly diagnostic mode with saved-capture
replay validation; it was not to replace canonical CTP automatically. The
captured chat does not establish a completed large QA result for this grouped
candidate. A proposed follow-up would capture attention from the first
generated answer token after BTP+QTP without prefill CTP, record comprehension
norm at layer input and output, and analyze current, aggregate-logit, and
grouped-head scores offline. The compacted context does not clearly establish
whether that exact timing experiment was completed, so a future agent must
verify artifacts rather than assume it ran.

### Interpretation of the paper's flat quality curve

The paper showed attention thresholds changing from `0.1` to `0.7` with only
small EM/F1 changes. The discussion inferred several compatible explanations:

- the benchmark has a broad quality plateau;
- many removed/retained visual tokens are redundant;
- answer-critical tokens survive across the threshold range;
- decoder throughput changes modestly because only part of total work is
  affected; and
- thresholds may change token quantity more than answer-relevant identity.

This curve is a useful implementation fingerprint but cannot by itself reveal
which attention tensor or aggregation rule the authors used.

## Decoder parity and intervention boundary

The user asked what a "fixed-layer sweep" and forced layer meant and why it was
necessary. The explanation was:

- native DocPrune selects the pruning layer using a comprehension threshold;
- a layer-dependent random/oracle/information-horizon experiment must compare
  every method at the exact same specified boundary;
- `B_K` means block `K` has executed with the full state/cache, then pruning is
  applied before block `K+1`;
- without a forced-boundary hook, methods could prune at different layers and
  the comparison would confound selection quality with pruning depth.

Tests defined `B_input` and boundaries after blocks `0`, `6`, `13`, `20`,
`23`, and `26`. They verified that blocks through `K` retain full caches,
later blocks see compacted inputs/caches, nonvisual tokens remain, original
token order and M-RoPE positions are preserved, and `M=0`, `M=|V|`, empty
visual populations, and comprehension failure-to-cross behave explicitly.

The seven-boundary all-kept parity gate was admitted by L40S job `62265662`
from clean runtime commit `36cb771...`. Earlier parity work found small BF16
logit differences on L40S, but full generated-token equality held; the final
tolerance was bounded using the reproduced maximum without relaxing behavioral
parity.

This was the critical cleanup that made the system acceptable as a controlled
baseline: with all tokens kept, the manual decoder behaves like stock
generation, so later differences can be attributed to the intervention.

## Approved random, coverage, removal, and attribution plan

The user supplied and approved a corrected experiment plan based on six papers:

1. [DocPrune](https://arxiv.org/abs/2604.22281)
2. [Wen et al., Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?](https://arxiv.org/abs/2502.11501)
3. [Zhang et al., Beyond Text-Visual Attention](https://arxiv.org/abs/2412.01818)
4. [Wang et al., When Token Pruning Is Worse than Random](https://arxiv.org/abs/2512.07580)
5. [Endo et al., Feather the Throttle](https://arxiv.org/abs/2412.13180)
6. [ContextCite](https://arxiv.org/abs/2409.00729)

The intended question was whether meaningful selection opportunity exists
beyond DocPrune attention: does random pruning already match it, does coverage
explain random's strength, is there an intervention-derived oracle gap, and at
which layers can visual-token removal still affect document QA?

### Source-first rule and pinned repositories

Task 1 required inspecting paper-author code before writing local equivalents.
Pinned sources included:

| Repository | Revision | Role |
|---|---|---|
| [Information-Horizon](https://github.com/YahongWang1/Information-Horizon) | `75909b2936a13d7214e83514f0bc0cb9cc91139e` | Wang random and conditional standalone-deletion source |
| [context-cite](https://github.com/MadryLab/context-cite) | `c11f8ace6e68ba0121b2e2f1f5c896da9e4156f4` | Sparse surrogate source |
| [MinerU](https://github.com/opendatalab/MinerU) | tag `mineru-3.1.0-released`, commit `d9cd58add047c2364c1198eefcb1ee9cd63a971a` | Independent layout-region tool |
| [VisPruner](https://github.com/Theia-4869/VisPruner) | `aefa01adc7c7ce6334e880c88225e90cede760d1` | Read-only diversity reference |
| [FEATHER](https://github.com/markendo/FEATHER) | `c2a09b2765967601054c7b1fd513ac3e94ee4fc9` | Read-only coverage reference |
| [FastV](https://github.com/pkunlp-icler/FastV) | `d1659729b5bf1be225e99ee15783deeea80f63b1` | Read-only Wen-studied baseline |

The audit found no author repositories linked for DocPrune or Wen; adjacent
code could not be presented as their implementation.

### Wang semantics correction

An external review challenged the initial interpretation of Wang's removal
test. The paper description suggested zero masking, but the pinned released
`cal_info`/`info_prune` code physically deletes 2x2 visual-token windows. The
user approved a dated amendment before affected Task 2/3 outcomes:

- faithful released-code Wang diagnostic = physical deletion of 2x2 windows;
- zero masking = separate local diagnostic with a different estimand.

The Wang random core is uniform sampling without replacement. LLaVA-specific
constants and layer conventions require explicit Qwen/DocPrune adaptation.

### Random and coverage controls

Task 4 implemented, test-first:

- deterministic literal-score top-M;
- deterministic aggregate-score top-M;
- global uniform random without replacement;
- page-stratified random matching aggregate top-M per-page counts;
- normalized per-page 4x4 grid-stratified random;
- coverage-matched identity shuffle preserving aggregate score's exact
  page-by-cell occupancy.

Every method restores retained tokens to original sequence order. Seeds are a
hash of experiment version, QID, boundary, policy, and repetition. Geometry is
computed after the actual BTP+QTP mask using the real per-page Qwen merged
grid; static factory-time geometry was forbidden.

Task 5 separated:

- literal/current native-threshold CTP;
- aggregate-logit native-threshold CTP;
- literal-score top-M;
- aggregate-score top-M;
- random/coverage policies using aggregate-native's exact `M`; and
- descriptive fixed-retention arms at 55%, 65%, and 80%.

Native threshold and forced top-M methods must never be mislabeled or silently
merged. Where there is no threshold-boundary tie, aggregate native and
aggregate top-M must select exactly the same mask at matched `M`; tied sets and
deterministic differences must otherwise be recorded.

### Visual-state removal curve

What the user originally called Wang's "information horizon" was narrowed to
an explicit visual-state removal/dependence curve unless near-zero standalone
token information is separately reproduced. Planned fixed boundaries were
`B_input`, `B_0`, `B_6`, `B_13`, `B_20`, `B_23`, and `B_26`, with BTP+QTP/no
CTP as the paired reference and all visual states physically deleted at the
selected boundary. A native `B_l*` diagnostic remained separate.

The plan called for simultaneous one-sided bounds and persistent-boundary
logic; non-monotonicity must be reported rather than forcing a single horizon.

### ContextCite and MinerU correction

The user initially proposed ContextCite as an answer-conditioned token oracle
using the correct-answer logit. The corrected plan made several distinctions:

- vanilla ContextCite attributes contribution to a specified response under a
  subset intervention distribution; it is behavioral attribution, not evidence
  ground truth;
- applying it to intermediate visual states is an adaptation, not a faithful
  reproduction of vanilla input-context ContextCite;
- MinerU is not used by the cited papers and must be set up independently;
- MinerU regions are page-space segments, not deep model tokens;
- when a MinerU text/layout segment is ablated at depth, the implementation can
  only mask or delete the visual tokens whose footprints fall within that
  region;
- whole-region costs require a knapsack/budget treatment and residual regions;
- if ContextCite proved infeasible, Wang's standalone contribution method was
  only a separately approved conditional fallback, not an automatic switch.

Pinned ContextCite defaults identified by source audit included 64 ablations,
keep probability `0.5`, Lasso `0.01`, intercept handling, and random-state and
response-target semantics. Pinned MinerU model revision was
`MinerU2.5-Pro-2604-1.2B` at
`d3f5e08d073c21466bbabe21c71bb1e9c2e595da`.

## Statistical and cohort rules

The 16-, 64-, and 245-question cohorts were all treated as development data,
not confirmatory holdouts. The remaining 181 questions in the 245 expansion
could not be relabeled as confirmatory after related outcomes had been seen.

Task 2 implemented:

- deterministic outcome-blind cohort ordering using
  SHA-256(`docprune-random-coverage-v2 || qid`);
- support-document connected components for clustered inference;
- primary aggregate-score-minus-global-random F1 estimand;
- equivalence margin `delta=1.0 F1`;
- TOST 90% interval and two-sided 95% superiority interval;
- nested mask resampling and document-cluster bootstrap;
- 100,000 terminal draws;
- Holm-adjusted secondary families; and
- fail-closed atomic sealing for QIDs, order, pages, features, source metadata,
  required N, and hashes.

Terminology was fixed to `equivalent`, `superior`, `inferior`, or `unresolved`.
Equivalence requires the full 90% TOST interval inside `[-1,+1]`, even if it
contains zero. A superiority interval containing zero is not evidence of
equivalence.

For the Task 6 developmental matrix:

- the 64-QID cohort is the first calibration gate;
- each random/coverage native-`M` policy uses repetitions `0..9`;
- if random-mask MCSE exceeds `0.25 F1`, add repetitions `10..19`;
- fixed 55/65/80% sensitivity uses three random repetitions per policy,
  reflecting Wang's reporting convention; and
- holdout membership is sealed only after developmental variance, required N,
  and final repetition count are known.

## Three canonical experiment documents

At the user's request, the original worktree created an all-in-one durable
experiment area:

`docs/experiments/docprune_random_oracle_horizon_2026-08-26/`

It contained:

- `EXPERIMENT_PLAN.md`: approved scientific design, adaptations, claim
  boundaries, and dated amendments;
- `IMPLEMENTATION_PLAN.md`: ordered tasks, status, gates, and next action;
- `EXPERIMENT_LOG.md`: source pins, jobs, artifacts, failures, results, and
  decisions.

Repository `AGENTS.md` was updated to require maintaining them. The experiment
log was not intended to contain every unrelated repository experiment. It
contained this plan's work plus prerequisite DocPrune baselines and diagnostics
needed to interpret it. Other historical experiments remained distributed
through SOL status, reproduction handoffs, and other project-specific logs.

An attached external review was read and converted into proposed change
"packages"; the user approved them. The attachment path in the chat was
`/home/lmalveau/.codex/attachments/f44b1afe-d3fe-432c-a872-14011513a3cd/pasted-text.txt`.
Its complete contents are not preserved in the compacted conversation, so this
summary cannot reproduce every review item. The resulting major corrections
are reflected above: no 245-question confirmation, no token-level ContextCite
oracle, formal equivalence, coverage controls, a renamed visual-state
dependence curve, and explicit faithful-versus-adapted labels.

## Task 6 implementation and portability smoke

### Fixed-page runtime and sealed inputs

Task 6 created a fixed-page path that reads questions/answers only from an
authenticated 64-QID JSONL and validates source/feature bytes only for the
selected QID. It does not construct the global dataset, enumerate all PDFs,
compute a corpus-wide source-order hash, or load the FAISS/global embedding/
token-map artifacts. The only live query operation is the pinned ColPali query
encoder; it does not retrieve or reorder pages.

Sealed no-replace inputs were:

- fixture v2:
  `/scratch/lmalveau/docprune/task6-fixed-page-gate-v1/fixture-stage64-top4-v2.json`,
  SHA-256 `32b3ddd6a1f608db509f002f9541dbc92317b59ac769f0b41fcb20bcc536652b`;
- gate v4:
  `/scratch/lmalveau/docprune/task6-fixed-page-gate-v1/gate-manifest-v4.json`,
  SHA-256 `4d44e297081b152c2e492ec68be58445c7ac17c8d106ceb2ddb1f6331fd57761`;
- eligible JSONL SHA-256
  `99d5e45a6ab55c298d8262cd23e996d78babf59cf45a453e3ba837fa40920795`;
- fixed-page reference SHA-256
  `381e19b033fa3e75e16d13bf6f13002bee7db2fc5102a9c6f374268131070642`;
- feature manifest SHA-256
  `ffa5979b3bf157adefcc132b0438af295ddafb2243377db4cdb8fcb8eaafe5da`.

The initial clean Task 6 runtime commit was
`b0c8742d358319b7b617b9b1d36ba1f5d1ea6c86`.

### First smoke failure and fix

First jobs were `62277597` A30, `62277598` A100-40GB, `62277599` H100, and
`62277600` L40S under
`/scratch/lmalveau/docprune/task6-smoke-b0c8742-v1`. All failed before the
first result because `AuthenticatedFixedPageRetriever.retrieve` called
`torch.as_tensor` on the one-element list returned by the real ColPali adapter
instead of unwrapping its `[tokens,128]` tensor. CPU fakes had returned a rank-3
tensor and missed the production interface.

The regression test was changed to reproduce the real list-of-one contract.
The retriever now validates batch length one and uses `encoded[0]`. Clean
successor commit was `8b02837fe58f952141d7abc3e95b5ec156840810`.

### Corrected four-GPU smoke

Corrected jobs and durations were:

| Job | GPU | Node | Duration | Result |
|---|---|---|---:|---|
| `62277904` | A30 24,576 MiB | `scg013` | 57 s | exit `0:0` |
| `62277905` | A100-SXM4-40GB | `scg004` | 42 s | exit `0:0` |
| `62277906` | H100 80GB HBM3 | `scg020` | 39 s | exit `0:0` |
| `62277907` | L40S 46,068 MiB | `scg017` | 47 s | exit `0:0` |

All used driver `595.71.05`, wrote exactly nine rows, and passed their
`artifacts.sha256` manifests.

The pinned analyzer wrote:

`/scratch/lmalveau/docprune/task6-smoke-8b02837-v2/analysis.json`

SHA-256:
`0bc408c47147cae316e45d334cff0793a66cc850d80f4c57be7d6b035270a508`.

Its canonical status was **`rejected`**, because the preregistered comparator
required exact input, selection, budget, geometry, cache, M-RoPE, and trace
identity across GPU families. Input identity passed in all comparisons, but
QTP diverged slightly:

| GPU | Post-BTP | Post-QTP | Literal native CTP | Aggregate native CTP | Aggregate retention |
|---|---:|---:|---:|---:|---:|
| A30 | 4,336 | 3,591 | 1,522 | 2,676 | 74.52% |
| A100-40GB | 4,336 | 3,597 | 1,508 | 2,673 | 74.31% |
| H100 | 4,336 | 3,589 | 1,507 | 2,692 | 75.01% |
| L40S | 4,336 | 3,586 | 1,527 | 2,689 | 74.99% |

Within every GPU:

- geometry count equaled post-QTP count;
- achieved budget equaled post-CTP count;
- all score/random/coverage arms matched aggregate-native `M`;
- every CTP arm used `B_14`;
- cache lengths matched retained sequences;
- retained M-RoPE shapes/digests were recorded; and
- aggregate-native and aggregate-score-top-M masks matched exactly.

The five deterministic cells all answered `Kom, Znojmo.` on every GPU. Random
answers differed because the upstream QTP populations and matched budgets
differed. The evidence therefore indicated real numerical threshold drift,
not retrieval drift, cache corruption, forced-boundary failure, or a mismatch
between aggregate native and top-M.

The user asked how large the difference really was. The answer distinguished
count from identity: the QTP count spread was only 11/~3,600 (`0.31%`), and
aggregate CTP count spread was 19 tokens (roughly `0.7%`), but the artifact did
not expose enough post-QTP original identities to claim an exact mask Jaccard.
One question cannot establish cross-hardware quality equivalence.

### Approved L40S amendment

The original frozen rule said the 64-QID matrix could proceed only if all four
GPUs had exact masks. Because L40S had always been designated canonical and
the others drift-only, the agent recommended a dated post-result amendment:

- L40S internal consistency is sufficient for Task 6 progression;
- A30/A100/H100 remain descriptive numerical-drift evidence;
- GPU families are never pooled for canonical quality/timing;
- the rejected analyzer result remains permanently rejected.

The user explicitly approved. The user also allowed another GPU family if
L40S waiting became excessive. The safe interpretation recorded in the plan
was to switch the **entire** canonical matrix to one alternative family under a
fresh handoff, preserving any partial L40S root as noncanonical. Mixing L40S
and alternate GPUs across QIDs would reintroduce hardware-dependent masks and
was not approved as the preferred analysis.

At the time, L40S availability appeared healthy: multiple public L40S nodes
were idle or partially available. Restricting to L40S might increase queue
time, but the sub-1% token-count difference itself would not materially change
per-question compute time.

## End-of-chat Task 6 launch state

The intended native matrix was:

- 64 independent one-QID shards;
- array `0-63%16`;
- one L40S, 4 CPUs, 96 GiB, 20 minutes per task;
- 45 generations per QID: five deterministic cells plus four randomized
  policies times ten repetitions;
- 2,880 total generations;
- fresh root proposed as
  `/scratch/lmalveau/docprune/task6-native-l40s-2d2bea2-v1`;
- no sensitivity or holdout work in the same submission.

The first handoff draft received independent **NO-GO before submission** for
three reasons:

1. fresh-root creation and `sbatch` were separate commands, so an existing-root
   failure did not necessarily block submission;
2. the handoff forbade resume, but the launcher automatically passed
   `--resume` for an existing shard directory; and
3. the run config, feature manifest, committed config, and Python executable
   were not all immutably pinned at launcher preflight.

These were corrected test-first in the clean runtime:

- successor commit:
  `2d2bea2722774e29009269637aa225131f170fd4`;
- L40S launcher SHA-256:
  `b2fa3193f9c9d3f06b2caef2873f5680a57b26e6842b2cde6fcdcbf599c94832`;
- matrix runner SHA-256:
  `54a6199a3d702032cf396a1bf85d257843ba8c3102d10ba280cc87c5459a53ae`;
- run-config SHA-256:
  `b9a6668aaf70c6059b175d18e29bc0082f76231d233a4d993b93fcb55ebbbb8f`;
- feature-manifest SHA-256:
  `ffa5979b3bf157adefcc132b0438af295ddafb2243377db4cdb8fcb8eaafe5da`;
- committed config SHA-256:
  `82463d2ef3296a199521f3f637b256341aad7938eb199cb55c6249debfea44aa`;
- Python fixed to `/home/lmalveau/mamba-envs/docprune-sol/bin/python`;
- existing shard directories rejected before runtime access;
- no automatic `--resume`.

Focused new tests passed `2/2`; the relevant suite passed `112/112`; the full
accepted suite passed `622`, with two opt-in real-model tests skipped and two
known stale-authority tests deselected. Ruff, shell syntax, and whitespace
checks passed.

The proposed corrected handoff was:

`sol/handoffs/DOCPRUNE_TASK6_NATIVE_L40S_64_2026-08-27.md`

However, the captured conversation does not contain a final re-review GO or a
returned Slurm job ID. A re-review subagent was started, then the turn was
interrupted by the user's request for an ELI5 summary. Therefore this chat does
**not** establish that the 64-QID array was submitted. Later Task 9 worktrees
strongly suggest subsequent progress, but that must be verified from current
repository/SOL records rather than inferred.

## ELI5 explanation given to the user

The simple framing was:

- document pages are thousands of visual puzzle pieces;
- BTP removes blank/background pieces;
- QTP removes pieces apparently unrelated to the question;
- CTP removes more pieces based on the model's attention;
- the 245-question test showed the largest quality loss when CTP was added;
- the new experiment compares attention-selected pieces with random and
  spatially balanced pieces at the same layer and exact budget;
- if random performs similarly, attention may not be identifying uniquely
  useful pieces; if aggregate attention wins, there is a real selection signal.

The four-GPU difference was described as tiny numerical rounding near hard
thresholds: different GPUs retained 3,586--3,597 QTP tokens. Using one GPU
family makes the controlled comparison clean. Queue waiting may increase, but
the token-count spread itself implies sub-1% compute impact and is smaller than
ordinary timing noise.

## Current conclusions versus superseded interpretations

### Current conclusions at chat end

- The baseline is suitable for controlled method comparison because all-kept
  manual generation matches stock behavior at tested boundaries.
- Current/literal CTP is a useful paper-textual reference but over-prunes and
  causes a controlled quality loss.
- Aggregate-logit CTP is the best-supported local reference from development
  fingerprints and small QA diagnostics.
- Neither reconstruction can be claimed as the official author
  implementation without author code or clarification.
- Compare a new method against both literal/current and aggregate-logit
  reconstructions under identical cached pages, BTP/QTP masks, decoding, and
  hardware.
- Current QTP remains the baseline after its candidate alternatives failed the
  245-question expansion.
- Cross-GPU exact threshold masks are not required after the approved L40S
  amendment, but GPU families must not be pooled.

### Superseded or rejected interpretations

- "Everything stated in the paper was copied, so the implementation must be
  faithful" was rejected as too strong; unspecified semantics are decisive.
- Treating CTP's discrepancy as only a threshold-scale problem was rejected;
  restoring retention count did not restore quality.
- QTP-mean, QTP-all, BTP-all, max-head CTP, literal sum-before-softmax, and
  original-token scaling were rejected as baseline replacements.
- Calling the planned fixed-depth removal sweep a faithful Wang information
  horizon was narrowed; it is a visual-state dependence curve unless the
  standalone-information condition is separately reproduced.
- Treating ContextCite/MinerU regions as token-level oracle ground truth was
  rejected.
- Requiring bit-identical masks across A30, A100, H100, and L40S was amended
  after being transparently recorded as a failed preregistered gate.

## Unresolved questions

1. What exact attention tensor, head aggregation, normalization domain, and
   threshold scaling did the DocPrune authors use?
2. Does aggregate-logit CTP outperform matched random and coverage controls on
   the complete 64-question developmental matrix?
3. Is its advantage, if any, larger than random-mask Monte Carlo uncertainty?
4. Does MCSE require expanding from 10 to 20 masks per random policy?
5. What fixed-retention behavior appears at 55%, 65%, and 80%?
6. What holdout sample size is required for `delta=1 F1` equivalence testing?
7. Was the 64-QID Task 6 matrix subsequently launched or completed outside
   this chat, and what did later Task 7--9 work establish?
8. Was the proposed first-generated-token/layer-input-versus-output attention
   timing diagnostic completed?
9. Did the grouped-query/KV candidate receive a live QA expansion?

## Safe continuation checklist

1. Read current authority files; do not execute from this historical summary.
2. Locate the later canonical Task 6--9 experiment documents and SOL handoffs.
3. Check Slurm/accounting and immutable scratch manifests for a Task 6 native
   matrix job ID before considering any resubmission.
4. If Task 6 never launched, reconstruct or locate clean commit `2d2bea2...`,
   verify all pins and that the proposed output root is absent, obtain a fresh
   independent GO, update current SOL authority, then submit exactly once.
5. If it completed, authenticate all 64 shards before aggregating outcomes.
6. Compute native-policy MCSE first. Run repetitions 10--19 only if MCSE is
   greater than `0.25 F1`.
7. Run fixed-retention sensitivity only in a separate no-replace root and under
   its own exact authority.
8. Compute developmental variance, equivalence power, required holdout N, and
   final repetition count before sealing the holdout.
9. Keep literal/current CTP, aggregate-logit CTP, deterministic score top-M,
   random, and coverage arms explicitly labeled and unmerged.
10. Never run retrieval, global-index loading, or feature reconstruction for
    these controlled fixed-page comparisons unless the user explicitly changes
    scope.

## Context gaps

- The conversation was compacted several times. This summary preserves the
  compacted state supplied to the agent, but not every intermediate response
  verbatim.
- The full contents of the pasted external review attachment are unavailable
  here; only its accepted consequences are preserved.
- Some exact artifact paths and full SHA-256 values for intermediate CTP
  diagnostics were not present in the final compacted state.
- The original working tree and its uncommitted files were unavailable on
  2026-09-05, so no claim is made that the historical files still exist at
  their old paths.
- Later Task 9 worktrees exist, demonstrating newer state outside this chat.
  Their results were deliberately not researched or incorporated because the
  user requested a summary of this conversation without new research.
