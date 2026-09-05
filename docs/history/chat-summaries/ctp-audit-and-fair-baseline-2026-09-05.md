# DocPrune CTP Audit and Fair-Baseline Chat Summary

This is a historical conversation summary, not active execution authority. It
covers work performed mainly on 2026-08-24 through 2026-08-31 in the former
worktree `/home/lmalveau/DocPrune-fix-evaluate-measurement`. At summary time
(2026-09-05), that worktree was no longer present and the active checkout was
`/home/lmalveau/DocPrune`, whose `agent-context/CURRENT_TASK.md` described a much
earlier reproduction stage. A future agent must reconcile commits and artifacts
against current authority before resuming or submitting jobs.

## Key takeaways

1. The completed local top-4 benchmark did not reproduce the paper's positive
   CTP effect. In a sealed 245-question fixed-page diagnostic, BTP+QTP scored
   F1 `43.6327`, while the literal/current CTP reconstruction scored `41.3347`,
   a paired change of `-2.2980` with a bootstrap interval
   `[-4.2980, -0.6286]`. This localized the statistically supported controlled
   loss to CTP rather than retrieval, BTP, or QTP.
2. The strongest CTP ambiguity is how the attention quantity is obtained and
   normalized. The canonical code averages post-softmax attention across heads
   and multiplies the result by the post-QTP visual-token count. The paper says
   to threshold "attention weights" but does not specify that multiplication,
   head aggregation, softmax domain, timing, or GQA handling.
3. A prompt-prefill aggregate-logit reconstruction—mean query-key logits across
   heads, visual-only softmax, then post-QTP-count scaling—best matched the
   paper-derived retention fingerprint and improved F1 by `+1.0816` over the
   current reconstruction. It still trailed BTP+QTP by `-1.2163`, so it is the
   best-supported local reconstruction, not identified author code.
4. The manual DocPrune generation path had a real control bug: stock Qwen used
   EOS IDs `[151645, 151643]`, while the manual decoder used only `151645`.
   This was fixed, recorded in run identity, and guarded at model load.
5. A final all-kept admission gate showed that stock Qwen and the manual
   DocPrune decoder receive exactly identical processor tensors and generate
   exactly the same complete token suffix on A30, A100-40GB, H100, and canonical
   L40S hardware. L40S required a documented first-step-logit absolute tolerance
   of `0.07` because of reproducible BF16/FlashAttention rounding; exact output
   tokens remained unchanged.
6. The resulting baseline is suitable for an internally controlled comparison
   of CTP methods, but not for claiming identity with the authors' unpublished
   implementation. The intended comparison has four arms: BTP+QTP without CTP,
   literal/current CTP, aggregate-logit CTP, and the new method.
7. No results for the user's own new CTP method were produced in this chat.
   Therefore the sentence "Our method outperforms the literal and
   best-supported reconstructions of DocPrune" was discussed only as a
   permissible future claim if paired results support it; it is not currently a
   verified result.
8. The first matched-budget random comparison was motivating but inconclusive:
   on 64 development questions, aggregate-logit CTP scored F1 `39.921875` and
   20-draw global random scored `39.427344`, a difference of `+0.494531` with a
   95% superiority interval `[-1.9008, +3.2469]`.
9. A prospective maximum-available holdout was later approved and completed on
   all 1,213 untouched authenticated single-hop questions. A fast preliminary
   pass gave aggregate-logit CTP F1 `45.97444`, random F1 `44.64373`, and a
   difference of `+1.33071`. This is not yet the canonical result: exact union
   admission and the prespecified 100,000-draw nested bootstrap were not
   completed in the preserved chat state.
10. The 1,213-question run contains only aggregate-logit CTP versus matched
    global random after the same BTP+QTP masks. It contains no BTP+QTP/no-CTP
    arm and no coverage-matched arm, so it cannot answer whether CTP itself
    improves over BTP+QTP.
11. Repository consolidation introduced a provenance risk. The three experiment
    trackers were absent from consolidated `main` when checked, and the last
    known surviving branch copy of the experiment log did not contain the final
    1,213-question preliminary entry. The present chat summary preserves that
    entry, but artifacts and repository history must be reconciled before it is
    treated as canonical.

## Starting state and operating constraints

The user directed the agent to read, in order, the dated reproduction handoff,
its reference file, `sol/CURRENT_SOL_TASK.md`, and relevant `AGENTS.md` files,
then continue without repeating completed work. The handoff said the full
top-4 benchmark was complete and identified CTP over-pruning as the main open
issue. The working branch initially pointed at
`a53bf95b8e4288f180553f04b4fa1eec8a841df5` (`fix: preserve result
classification in quality merge`). Existing artifacts and dirty documentation
were to be preserved.

The user repeatedly imposed these experiment rules:

- Use short GPU jobs and several GPU sizes to diagnose quickly.
- CPU analysis can run directly on the lightwork CPU allocation; it does not
  require a submitted job.
- For controlled diagnostics, reuse cached ordered retrieved pages and persisted
  page features. Do not perform fresh/global retrieval unless explicitly asked.
- Preserve all historical scratch roots, failed jobs, clean runtime checkouts,
  and uncommitted changes.

The no-retrieval rule was added to both repository and SOL `AGENTS.md` files in
the old worktree after the agent accidentally discussed retrieval ambiguously.
The active `/home/lmalveau/DocPrune/AGENTS.md` observed on 2026-09-05 did not
contain that added wording, so future agents must not assume it was merged.

## Verified benchmark and diagnostic evidence

### Controlled 245-question stage study

The sealed top-4 study held question IDs, ordered retrieved pages, answer model,
prompt, and evaluator fixed. Reported F1 values were:

| Stage | F1 |
|---|---:|
| All-kept | `44.8612` |
| BTP | `43.1143` |
| BTP+QTP | `43.6327` |
| BTP+QTP+current CTP | `41.3347` |

The adjacent BTP and QTP changes were not statistically conclusive. The CTP
change was `-2.2980` F1 with interval `[-4.2980, -0.6286]`, making it the only
stage with a supported negative controlled effect.

The full benchmark also retained a retrieval-identical stratum of 2,126 out of
2,441 questions. That stratum still lost `-1.1980` F1 with an interval excluding
zero, demonstrating that retrieval-page differences could not explain the
direction of the controlled CTP loss.

The local final visual-token drop was about `81.72%`, compared with the paper's
reported `74%`, which was consistent with over-pruning. Exact equivalence of the
paper's drop-accounting endpoint was not proven.

### Paper sensitivity-table interpretation

The user supplied the paper's threshold-sensitivity table. Across attention
thresholds `0.1`, `0.3`, `0.5`, and `0.7`, throughput changed modestly while EM
and F1 stayed nearly flat. The discussion concluded that this may indicate
answer redundancy, stability of the selected ranking, or a broad threshold
plateau. It does not uniquely reveal the authors' attention-score formula.
Matching the whole retention/throughput curve was considered a stronger
fingerprint than matching one retention percentage.

## Code-versus-paper audit

The user challenged an earlier implication that everything in the paper had
been copied. The agent and three read-only audit subagents then audited the
executable code rather than relying on prior conversational claims. The audits
used the local DocPrune main PDF and supplement where available. No code or jobs
were changed by the audit subagents.

### BTP: strong algorithmic fidelity, with implementation omissions

Verified paper-aligned behavior included:

- BTP runs before both retrieval and QA visual encoders.
- It operates on Qwen/ColPali-compatible 2-by-2 spatial merge groups.
- The implementation converts patches to grayscale, finds the page-wide mode,
  computes the fraction of pixels satisfying strict
  `abs(pixel - mode) < tau_e`, and retains patches with
  `background_ratio <= tau_bg`.
- The strict and inclusive boundaries were explicitly tested.

Local choices not fixed by the paper included BT.601 grayscale coefficients,
float/round/uint8 arithmetic, mode tie behavior, processor-space patch size,
and reducing four patch decisions into one 2-by-2 group. The default group
reduction was `any`.

CPU replays reduced concern about several of these choices:

- BT.601 truncation changed 49 groups and channel mean changed 50, while overall
  retention stayed near `52.2%`.
- An inclusive pixel boundary changed 989 groups but moved retention only from
  `52.2453%` to `51.6292%` with Jaccard `0.9880`.
- BTP `mean` reduction changed every mask (Jaccard `0.5887`) but was neutral on
  the 12-question calibration set.
- BTP `all` retained only `6.93%` and was rejected.

### QTP: core equations reproduced, details remain underspecified

Verified paper-aligned behavior included:

- Reuse of retrieval document embeddings and question embeddings.
- Cosine normalization and summing similarity over question-token rows.
- Bilinear resize from retrieval-map resolution to Qwen-map resolution.
- Gaussian smoothing after resize.
- Inclusive retention `score >= tau_qst` and intersection with BTP survivors.

Unspecified local choices included which prompt/special tokens count as
question tokens, row-major 32-by-32 ColPali layout recovery, bilinear
`align_corners=False`, Gaussian `sigma=1.0`, finite support, replicate padding,
2-by-2 `any` reduction, and handling sparse holes or empty masks.

The most important direct paper conflict was the checkpoint: the paper and
supplement named ColPali-v1, while the local configuration pinned
`vidore/colpali-v1.2`. This affects global retrieval and QTP embeddings even
when page identities are fixed. It does not invalidate an internally controlled
CTP comparison if all arms share the same frozen BTP/QTP masks, but it weakens
claims of exact paper reproduction.

QTP `mean` reduction looked positive on 12 calibration questions but failed the
245-question expansion: F1 changed from `43.6327` to `42.6653`, paired
`-0.9673`, interval `[-3.1755, +1.0408]`. QTP `all` lost `8.3333` calibration
F1. Neither was promoted.

### CTP: the primary unresolved author ambiguity

Paper-explicit behavior reproduced at a high level included:

- Select the first decoder layer whose last-token hidden-state L2 norm crosses
  the comprehension threshold, using an inclusive boundary.
- Recompute only the final query against keys at the selected layer rather than
  materializing full attention.
- Apply one visual-token CTP selection, retain tokens satisfying an inclusive
  attention threshold, leave the selected layer's cache full, and compact deeper
  layers.

The canonical operand was not literal paper attention. The code computed:

`mean_heads(softmax_over_all_keys(q_last * K / sqrt(d)))[visual]`

and then multiplied it by the number of post-QTP visual tokens before applying
the paper threshold. The paper does not specify that scaling. Other missing
semantics included:

- prompt-prefill last token versus a generated answer token;
- selected-layer input versus output as the comprehension boundary;
- pre-softmax logits versus post-softmax weights;
- softmax over all keys versus visual keys only;
- mean, sum, maximum, selected heads, or GQA-group aggregation;
- original versus post-BTP/QTP token-count scaling;
- cache compaction boundary, generated cache positions, and M-RoPE positions;
- precision and equivalence to the authors' attention backend.

The factory also failed to pass configured `reconstruction_defaults` into the
production answerer. Defaults happened to match canonical behavior, so canonical
runs were not changed, but alternate manifests could claim settings that were
not actually executed. This was classified as a local control-surface defect.

### Evaluation and system fidelity

The metric implementation matched the pinned upstream M3DocRAG evaluator in
local checks, including normalization, list matching, numeric handling, EM, and
F1. Dataset rendering, prompt bytes, Qwen revision, and decoding cap were not
published precisely by DocPrune and therefore remained inherited reconstruction
choices.

Efficiency results were not paper-equivalent: the paper used RTX A6000, while
the local quality studies used mixed newer GPUs, and FLOP profiling was disabled
by default. Quality conclusions should not be presented as paper throughput or
TFLOP reproduction.

## CTP candidates considered and tested

The conversation considered direct probability thresholds, head sums, raw
pre-softmax logits, visual-only softmax, full-key softmax, maximum heads,
GQA-aware grouping, different scaling bases, and prompt-versus-generated-token
timing. A direct raw-logit threshold was plausible numerically but considered
less likely because the paper called the values "attention weights."

### Prompt-prefill aggregate-logit candidate

The strongest retained candidate averaged raw query-key logits across heads,
then applied softmax only over visual keys and multiplied by the post-QTP visual
count. Its sealed 245-question analysis was:

`/scratch/lmalveau/docprune/ctp-aggregate-logit-stage245-fixed-v1/analysis-final.json`

Reported results:

- F1 `42.4163`.
- `+1.0816` versus literal/current CTP.
- `-1.2163` versus BTP+QTP.
- Retention `67.2929%`, near the paper-derived top-4 target around `65%`.

This was encouraging and the best local retention/quality compromise, but not
statistically conclusive evidence of author semantics. It did not reproduce the
paper's positive CTP contribution.

### Rejected or non-promoted candidates

- Original-scale attention reached about `58.01%` retention but harmed
  calibration F1.
- Maximum-head aggregation was too permissive and harmed quality.
- GQA grouping averaged the seven query heads sharing each of Qwen's four KV
  heads, softmaxed within groups, and averaged the four maps. A CPU fingerprint
  error of about `1.72` percentage points beat aggregate-logit's earlier
  `2.50`, and masks differed by roughly `9%`. However, the 16-question live QA
  gate scored F1 `43.50`, equal to BTP+QTP and below current CTP's `46.8333` on
  that calibration subset. It was rejected. The method was architecture-based,
  not specified by the paper; the discussion noted that such dependence could
  make a method architecture-sensitive but not necessarily invalid.
- First-generated-token, layer-output aggregate-logit scoring matched the
  retention target exceptionally well: `65.0360%` versus `65.0000%`. In the
  16-question paired QA gate it produced answers byte-for-byte identical to
  BTP+QTP on all 16 examples. Current prompt CTP changed four answers and gained
  partial F1 on one calibration example. The candidate had zero wins, 11 ties,
  one loss versus current on the 12 calibration examples, with paired F1
  `-3.3333` and interval `[-10.0, 0.0]`. It was rejected.
- The first-generated layer-input interpretation, later generated-token timing,
  other GQA reductions, and deeper cache/position semantics remained untested or
  incompletely tested.

The generated-token capture analysis was stored at
`/scratch/lmalveau/docprune/ctp-generated-query-capture-v2/l40s/analysis.json`
(file SHA-256
`9e371af8177d11640b5635f61be750ce28ac00143bac305d4702e741ac621700`).
The rejected live QA analysis was
`/scratch/lmalveau/docprune/ctp-generated-query-qa-v1/l40s/analysis.json`
(file SHA-256
`80573b02f2a3b450ad7020eb8c86e1ace30a02179e3db68ddb8a40e5b57dd073`).

## Fair-baseline generation fix

The user wanted a baseline suitable for developing a new CTP method, even if
exact author CTP identity remained unknowable. Before freezing that baseline,
the agent audited stock versus manual Qwen generation.

### Root cause found

For the pinned Qwen checkpoint
`eed13092ef92e448dd6875b2a00151bd3f7db0ac`:

- `generation_config.json` specified EOS `[151645, 151643]`.
- `config.json` specified only `151645`.
- Stock `model.generate()` inherited the generation-config list.
- `DocPruneQwenAnswerer` passed only `model.config.eos_token_id` into its manual
  loop.

Thus the pruned path could continue past token `151643`, creating answer changes
unrelated to pruning. Existing tests used no EOS IDs and compared only two new
tokens, so they could not catch this.

### Implemented corrections

Historical commits in the old worktree were:

- `a1c0a9ba5d2e5e0e842dc8e15a1db48b05a82162`: resolve one shared complete EOS
  set for stock and manual generation; record EOS IDs in run identity; reject a
  pinned-model EOS mismatch.
- `15336753c03a84e34328c5dad521220caa97a18e`: run the real-model gate with the
  production `flash_attention_2` backend.
- `5f350d73d9992f2914f585d843b538528bb7a14d`: check exact generated-token parity
  before diagnostic logit tolerance.
- `dd5f000a909a718826541df816a4b65396764e1c`: retain exact generation parity and
  admit the bounded L40S first-step logit drift with `rtol=0.02, atol=0.07`.

The real processor probe compared normal original-page processing with the
DocPrune prepare/reraster path and found exact equality:

- `input_ids` shape `[1, 2537]`, exact;
- `attention_mask`, exact;
- `pixel_values` shape `[10032, 1176]`, maximum absolute difference `0`;
- `image_grid_thw`, exact;
- source raster `(1224, 1584)` and smart-resized raster `(1232, 1596)`.

The final baseline protocol was written historically as
`docs/reproduction/FAIR_CTP_BASELINE_2026-08-26.md`. That file was not present in
the active checkout at summary time.

## GPU admission sequence

All jobs used one fixed cached page, no ColPali, no retrieval features, and no
global index. Each requested one GPU, four CPUs, 64 GiB, and at most 20 minutes.

### Superseded attempt

Jobs `62211245` through `62211248` were canceled while pending, with elapsed
`00:00:00` and no nodes assigned, after discovering that the gate had not
explicitly loaded the production FlashAttention-2 backend. They produced no
promotable result.

### Cross-hardware v2

Runtime `15336753c03a84e34328c5dad521220caa97a18e` produced:

| GPU | Job | Result | Time/node |
|---|---:|---|---|
| A30 | `62212255` | Pass | `00:00:27`, `scg013` |
| H100 NVL | `62212256` | Pass | `00:01:22`, `scg028` |
| L40S | `62212257` | Logit tolerance failure | `00:00:25`, `scg022` |
| A100-SXM4-40GB | `62212258` | Pass | `00:00:24`, `scg003` |

The three passing jobs established exact preprocessing, complete generated
suffix equality, and the original `0.02/0.02` logit tolerance. Their
`artifacts.sha256` file digests were:

- A30: `52f25ba9c28d50976d502fc419c830abc6a9329a4c7a2ef60adc6dbd37a22a7b`
- A100-40GB: `69f4581290689467f364a3e137b1c15e6ffe0c15a2ca204614142fd1d12c1ab9`
- H100: `166d2ebef05aa76673c9b2a62ab26ba47bef9cae4ff7416202d175d66271fc1a`

L40S differed beyond the original first-step-logit tolerance in
1,710/152,064 values (`1.1%`), with maximum absolute difference
`0.06640625`. Because the test asserted logits first, that run did not establish
whether generated suffixes matched.

### L40S discriminator and final admission

Job `62219487` on a second L40S node (`scg017`) moved the exact suffix assertion
before the unchanged logit assertion. The entire generated suffix matched
exactly, after which the same 1,710-element and `0.06640625` logit discrepancy
recurred. This isolated the issue as reproducible L40S BF16/FlashAttention
numerical drift without behavioral divergence.

Job `62219618`, runtime
`dd5f000a909a718826541df816a4b65396764e1c`, kept exact preprocessing and exact
full-generation requirements while changing only logit `atol` to `0.07`. It
passed on NVIDIA L40S (`scg017`) in 17 seconds. Artifact details:

- Root:
  `/scratch/lmalveau/docprune/qwen-all-kept-parity-dd5f000-v4-l40s/result`
- `artifacts.sha256` SHA-256:
  `84bda742007f3702dd752d536c8257022af89645540846243f5a79df52275d43`
- JUnit SHA-256:
  `a5d265cbd8ee13e8ac6221e60320b687985ed3016fe6ed532d47d382c133c9e6`
- Pytest log SHA-256:
  `801e1eb0dc7e74ece3624dd6a2c269021c5efd86a355306e3242891bd5318848`

Final clean-runtime CPU verification reported `442 passed`, one opt-in GPU test
skipped, and two known stale-authority documentation tests deselected. Ruff
passed. This admitted the local generation control; it did not prove exact
author CTP semantics.

## Intended comparison and claim boundary

The recommended paired experiment has four arms under the corrected generation
contract:

1. BTP+QTP with no CTP, as the no-CTP anchor.
2. Literal/current prompt-prefill CTP, closest to the paper's ordinary meaning
   of post-softmax "attention weights."
3. Prompt-prefill aggregate-logit CTP, the primary and empirically
   best-supported local DocPrune reconstruction.
4. The user's new CTP method as the only changed component.

All arms must reuse identical cached ordered pages, persisted page features,
BTP/QTP masks, Qwen checkpoint, processor tensors, prompt bytes, decoding,
cache/position behavior outside CTP, question cohort, and evaluator. Historical
pre-EOS-fix scores must not be directly compared with new-method results; the
reference arms must be rerun under the corrected contract.

If paired results support it, defensible wording is:

> The method outperforms the literal and best-supported local DocPrune
> reconstructions under matched experimental controls.

It is not defensible to say the method beats the authors' official unpublished
implementation. The user specifically asked how confident the agent was that
no small implementation mistakes remained. That question was interrupted
before a substantive answer. The evidence supports high confidence in the
all-kept generation control and reasonable confidence in the core BTP/QTP
equations, but not confidence that all author-specific details are correct.
Exact CTP parity is low-confidence because essential semantics were omitted;
ColPali-v1.2 versus v1 is a known direct mismatch; and small cache, position,
token-membership, smoothing, or block-reduction differences may remain. Any
future claim must reflect that calibration.

## Broader experiment program: proposal, correction, and approval

An initial request to evaluate a six-paper experiment plan was briefly deferred
while the all-kept GPU admission was checked. That deferral was temporary and is
superseded: the papers were subsequently analyzed, the plan was technically
corrected, and the user approved the corrected program.

The required papers were:

1. [DocPrune, arXiv:2604.22281](https://arxiv.org/abs/2604.22281)
2. [Wen et al., *Token Pruning in Multimodal Large Language Models: Are We
   Solving the Right Problem?*, arXiv:2502.11501](https://arxiv.org/abs/2502.11501)
3. [Zhang et al., *Beyond Text-Visual Attention*,
   arXiv:2412.01818](https://arxiv.org/abs/2412.01818)
4. [Wang et al., *When Token Pruning Is Worse than Random*,
   arXiv:2512.07580](https://arxiv.org/abs/2512.07580)
5. [Endo et al., *Feather the Throttle*,
   arXiv:2412.13180](https://arxiv.org/abs/2412.13180)
6. [ContextCite, *Attributing Model Generation to Context*,
   arXiv:2409.00729](https://arxiv.org/abs/2409.00729)

The approved scientific question was whether a useful token-selection
opportunity exists beyond DocPrune's attention signal: whether random selection
already matches it, whether an intervention-derived privileged ranking exposes
an oracle gap, and at which decoder layers selective visual-token pruning still
affects document-QA quality.

### Corrections made to the user's draft

- Random pruning is meaningful only when applied to the same post-BTP+QTP
  visual-token population, at the same decoder boundary, with the same exact
  per-question retained-token count and original token order as the DocPrune
  comparator. Generic random removal at an arbitrary layer would not be a fair
  control.
- A forced-layer diagnostic hook was required because ordinary DocPrune chooses
  a layer dynamically from the comprehension threshold. "After layer K" means
  block K has executed and populated its full cache, then the intervention is
  applied before block K+1. This is an explicit adaptation used to compare
  methods at identical depths; it is not the native DocPrune policy.
- Wang's all-visual-token-removal test should be treated as a layer-dependent
  visual-dependence or information-horizon diagnostic. Near a layer where
  removing every remaining visual state has little effect, differences among
  token rankings may naturally become unimportant.
- ContextCite supplies behavioral attribution to a specified response under an
  intervention distribution; it is not evidence ground truth. Applying it to
  intermediate visual states is an adaptation beyond vanilla input-context
  ContextCite.
- MinerU 2.5 Pro is independent of the cited pruning papers. It segments page
  regions. At a deep decoder layer, each region is represented by a mask over
  the visual tokens spatially assigned to that region; the experiment cannot
  substitute MinerU text segments for actual hidden-state tokens.
- The proposed privileged diagnostic scores the correct/target answer token
  under interventions, not merely the token the model generated. This makes it
  answer-conditioned and unsuitable as a deployable pruning policy.
- If ContextCite plus MinerU proved unusable, Wang's released importance
  implementation was an approval-gated fallback. It was not to be substituted
  silently.
- Code released by the papers must be obtained, pinned, and inspected before
  synthesizing local implementations. Adaptation is allowed for Qwen, dynamic
  visual populations, cache behavior, and the local evaluator, but the
  published implementation should be the starting point.

### Historical planning documents and source pins

Three all-in-one program trackers were created in the old worktree:

- `docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_PLAN.md`
- `docs/experiments/docprune_random_oracle_horizon_2026-08-26/IMPLEMENTATION_PLAN.md`
- `docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_LOG.md`

They respectively held the scientific protocol, sequential implementation and
execution anchor, and append-only experiment results. The relevant old
`AGENTS.md` files were amended to require updating these documents as results
arrived or the approved course changed. The active checkout did not contain
this directory when consolidation was checked.

Known code/source pins gathered during setup were:

- Information-Horizon repository commit beginning `75909b`;
- ContextCite repository commit beginning `c11f8a`;
- MinerU tag 3.1 commit beginning `d9cd58`;
- VisPruner commit beginning `aefa01`;
- FEATHER commit beginning `c2a09b`;
- FastV commit beginning `d16597`.

No DocPrune author repository was found. The compacted chat does not preserve
the repository URLs or complete hashes, so they must be recovered from the
historical implementation plan or primary papers rather than reconstructed from
memory.

## Matched-budget random controls

### Frozen method

For each question, BTP+QTP defines an eligible visual-token population `V`.
Aggregate-logit CTP runs with its native threshold and dynamic selected boundary
to determine a retained count `M`. Every matched comparator acts on the same
`V`, at that boundary, retains exactly `M` tokens, and preserves their original
sequence order.

The main random policy is global uniform sampling without replacement, matching
Wang's released sampler as closely as the Qwen/DocPrune adaptation allows.
Additional local diagnostics were defined:

- page-stratified random;
- grid-stratified random;
- coverage-matched random, which preserves aggregate CTP's per-page and
  normalized 4-by-4 cell counts while randomizing token identities inside each
  cell.

Coverage-matched random is an original diagnostic inspired by the spatial
coverage hypothesis. It is neither faithful DocPrune nor a method claimed by
Wen or Wang.

### Verified 64-question development result

The first native-M comparison used 20 deterministic random masks per question.
Ten masks had produced Monte Carlo standard error `0.560332` F1, above the
prespecified `0.25` trigger, so masks 10 through 19 were added to every random
policy. Attention was deterministic and evaluated once per question.

| Policy | F1 | Qualification |
|---|---:|---|
| Aggregate native/top-M | `39.921875` | Deterministic CTP |
| BTP+QTP, no CTP | `39.562500` | Development anchor |
| Literal native | `39.531250` | Own threshold and budget |
| Literal score top-M | `39.812500` | Aggregate-matched budget |
| Global random, 20 masks | `39.427344` | Primary random control |
| Coverage-matched random | `39.582031` | Local diagnostic |
| Page-stratified random | `38.702344` | Local diagnostic |
| Grid-stratified random | `38.637500` | Local diagnostic |

Aggregate CTP exceeded global random by `+0.494531` F1. The 90% TOST interval
was `[-1.5266, +2.7273]`; the 95% superiority interval was
`[-1.9008, +3.2469]`. Therefore the result was encouraging but officially
unresolved: random came close enough to motivate a larger study, but the sample
did not establish either equivalence or attention superiority.

The native aggregate retention averaged `66.0558%`. Literal native retention
was about `40.9280%`; consequently, the earlier 245-question aggregate-versus-
literal gain was confounded by budget. At aggregate's exact per-question `M`,
the two rankings differed by only about `0.11` F1.

### Fixed-retention result and caveat

Aggregate and literal top-M scores were nearly flat at 55%, 65%, and 80%
retention:

| Retention | Aggregate F1 | Literal F1 |
|---:|---:|---:|
| 55% | `39.890625` | `39.906250` |
| 65% | `39.937500` | `39.796875` |
| 80% | `39.890625` | `39.984375` |

This is not the paper's threshold sweep. Fixed top-M guarantees a fraction;
the paper's `tau_att` retains all tokens passing a score cutoff, so its budget
can differ across questions. Random fixed-retention masks were independently
resampled at each rate and used only three repetitions, making their apparent
curve noisy and unsuitable for causal interpretation across retention levels.

## Prospective 1,213-question primary holdout

### Why the design changed

The existing 16-, 64-, and 245-question cohorts had all become development
data. A power calculation estimated 2,199 questions for 80% power at the
prespecified +/-1 F1 equivalence margin, but only 1,213 untouched,
authenticated single-hop questions were available. Their estimated power was
`0.5799759174`.

The user approved a prospective maximum-available amendment: run all 1,213,
retain the original estimand, masks, and +/-1 margin, and classify a failed TOST
as unresolved rather than changing the hypothesis after seeing outcomes.
Multi-hop was not pooled into the primary to repair power. It could be a
separate secondary stratum using fixed cached pages. Fixing cached retrieval
does remove comparison drift, but top-4 retrieval can still omit evidence and
make multi-hop a different assay.

The holdout order was outcome-blind and deterministic:
`SHA256("docprune-random-coverage-v2" || qid)`. It was random-like but not a
formal stratified sample.

Frozen identity hashes recorded in the chat were:

- manifest-file SHA-256 beginning `3948fd7e` and ending `1b69`;
- fixture SHA-256 beginning `8ee103c8` and ending `103`;
- gate SHA-256 beginning `855bb148` and ending `3b2`;
- selection SHA-256 beginning `4c79d806` and ending `952`.

The compacted context does not retain the complete hash strings. The run used
only cached ordered pages and persisted features: no retrieval, global-index
loading, or feature rebuilding.

### Fair-share correction and batching

The original L40S Slurm array, job `62319253`, used one question per task over
indices `0-1212%24`, requested four CPUs, 96 GiB host RAM, and 20 minutes. The
user correctly objected that one shard per question and excessive host-memory
reservation were consuming fair-share inefficiently.

A quick measurement over completed shards found host `MaxRSS` peak/mean about
`1.965/1.929 GiB`. Thirty-six original QIDs, indices 0 through 35, completed
validly; the remaining pending tasks were held and canceled without discarding
the completed outputs.

A clean batched runtime was created at
`/home/lmalveau/DocPrune-task6-batched-runtime-20260828`, commit
`289247b7306...`. Its launcher was
`examples/sbatch/42_docprune_task6_holdout_batched_l40s.sbatch`, with file hash
beginning `2510b850`. It assigned at most 12 sequential QIDs to each of 99
array tasks, throttled to 12 concurrent tasks, with one exact L40S, four CPUs,
and 20 minutes per task.

A 6 GiB host request was rejected by cluster policy before creating a job
because GPU jobs required at least 24,000 MB. The request was corrected to 24
GiB. Replacement job `62319936` wrote to:

`/scratch/lmalveau/docprune/task6-holdout-primary-1213-v1/matrix-batched-v1`

All 99 replacement tasks completed with exit `0:0`, generally in 13-14
minutes; the final one-question task completed in 1:03. Together with the 36
original shards, this covered all 1,213 QIDs and the expected 25,473 rows.

Each question has 21 rows: one deterministic aggregate-logit CTP top-M result
and 20 global-uniform random masks. Despite earlier broader plans, this primary
contains no coverage arm, literal CTP arm, or BTP+QTP/no-CTP arm.

### Interim analyses and changing interpretation

The original plan prohibited reading partial-array outcomes to prevent optional
stopping, tuning, and accidental use of incomplete files. The user explicitly
requested preliminary analysis while the fixed run continued. Because all
1,213 questions were already committed, their SHA order was outcome-blind, and
no design changes were permitted, descriptive terminal-prefix analyses were
allowed and clearly labeled noncanonical.

At 252 terminal QIDs:

- aggregate CTP F1/EM: `47.1349 / 39.6825`;
- random-r20 F1/EM: `45.0839 / 37.0040`;
- paired difference: `+2.0510` F1 and `+2.6786` EM;
- first/second 126-question half differences: `+1.600` and `+2.502`;
- retention: `65.7883%`;
- random Monte Carlo standard error: `0.2476`;
- wins/ties/losses: `61/161/30`;
- descriptive 10,000-draw nested-bootstrap 95% interval:
  `[+0.3315, +3.7675]`, with 90% `[+0.5990, +3.4868]`.

Artifacts were:

- `/scratch/lmalveau/docprune/task6-holdout-primary-1213-v1/analysis-interim-252-v1.json`,
  SHA-256 beginning `b995f963` and ending `8373`;
- `/scratch/lmalveau/docprune/task6-holdout-primary-1213-v1/analysis-interim-252-ci-v1.json`,
  SHA-256 beginning `88e1ae68` and ending `f08`.

At 612 terminal QIDs:

- aggregate CTP F1: `46.6961`;
- random F1: `45.8154`;
- difference: `+0.8806`;
- approximate IID 95% interval: `[-0.1927, +1.9539]`;
- fast approximate nested 10,000-draw 95% interval:
  `[-0.2760, +2.0392]`;
- 90% TOST interval: `[-0.1085, +1.8302]`;
- first/second half differences: `+1.7709` and `-0.0096`.

This showed that the early +2 F1 estimate was favorable sampling fluctuation.
The user celebrated the near-equivalence trend, but that interpretation was
subsequently superseded by the full preliminary point estimate.

At all 1,213 questions, a fast preliminary scoring pass reported:

- aggregate-logit CTP F1: `45.9744435284`;
- global random-r20 F1: `44.6437345425`;
- paired difference: `+1.3307089860`;
- quick IID approximate 95% interval: `[+0.5506988, +2.1107192]`.

This makes the final observed advantage larger than at 612 questions and
weakens the hypothesis that global random is equivalent to CTP. It is still not
the canonical statistical result. A slow full validator was stopped after
runtime/output-capture problems, so exact union authentication and the
prespecified 100,000-draw nested mask/support-component bootstrap remained
unfinished. The quick IID interval does not substitute for that analysis.

The result also does not show that aggregate CTP improves absolute document-QA
quality. It only compares aggregate's selected token identities with global
random at the same dynamic layer and token count after identical BTP+QTP. A
same-holdout BTP+QTP/no-CTP arm is required to measure the incremental effect of
CTP itself.

## Other tasks and partially resolved work

- Task 7 all-visual-state removal fixed-grid and native-boundary arrays
  `62314816` and `62314817` completed on exact L40S with exit `0:0`. The chat's
  last reliable state did not include canonical admission and interpretation.
- Task 8 MinerU mapping was reportedly admitted. It partitioned 3,586 visual
  tokens into 95 region sources; associated job `62313385` was named. The
  precise admitted artifact and analysis were not retained in compacted
  context.
- Task 9 regional-attribution smoke job `62315446` completed on L40S. An
  A100-80 job `62315546` was canceled while pending without GPU use. Later Task
  9 work existed on another branch but was not analyzed in this chat summary.
- The original memory-heavy quality-only merger failed at both 32 and 64 GiB
  because deep validation loaded approximately 23.9 GB of embeddings, 2.8 GB
  of JSON token maps, and a 23.5 GB FAISS index. A lightweight quality-only
  validation path was implemented and earlier results were published, but the
  compacted chat does not retain enough artifact names or final values to
  reproduce that publication here. This is a context gap, not evidence that the
  work was absent.

## Repository consolidation and experiment-log provenance

The full-cohort preliminary point estimate was appended in the old checkout to:

`/home/lmalveau/DocPrune-fix-evaluate-measurement/docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_LOG.md`

The entry was titled `Task 6 full-cohort preliminary point estimate —
2026-08-31` and explicitly labeled the result noncanonical pending union
admission and the 100,000-draw nested bootstrap.

After branches were consolidated, that old checkout no longer existed. At the
time of inspection, consolidated `/home/lmalveau/DocPrune` did not contain the
three experiment trackers. A remaining tracked copy was located at:

`/home/lmalveau/DocPrune-task9-analysis-cpu-20260828/docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_LOG.md`

on branch `codex/task9-analysis-cpu-20260828`, then at commit beginning
`ef3202e`. That copy did **not** contain the final 1,213-question preliminary
entry. Thus the conversation/archive may be the only surviving record of that
entry until scratch artifacts are reauthenticated. Do not represent it as
merged or canonical merely because it appears here.

## Important conversational corrections and decisions

- The statement that everything in the paper had been copied was withdrawn.
  Core equations were implemented, but several runtime semantics are omitted
  by the paper and cannot be guaranteed.
- Pre-softmax aggregate logits are not confirmed author semantics. They are the
  best-supported tested reconstruction and the chosen primary internal
  baseline.
- Matching the paper's retention fingerprint does not prove quality fidelity.
  Generated-token visual-softmax matched retention almost exactly but did not
  improve over BTP+QTP in the small gate.
- The `+1.0816` aggregate-versus-literal result did not isolate ranking because
  native thresholds retained very different token counts. Matched-budget
  development results were nearly tied.
- Twenty repetitions apply only to stochastic random masks. They were a local
  variance-control extension, not a paper-reported setting.
- The 1,213-question primary is random-only versus aggregate CTP. It cannot be
  used to report a BTP-to-QTP-to-CTP trend or a coverage comparison.
- Cached retrieved pages and persisted features are mandatory for controlled
  work. Fresh retrieval or global-index loading must not occur unless the user
  explicitly requests it.
- Single-hop became the primary holdout because it supplied the untouched
  authenticated pool and avoids mixing a different retrieval/evidence regime.
  Cached pages make a multi-hop secondary comparison fair between policies,
  but do not remove missing-evidence limitations.
- The user requested short GPU jobs, modest host-memory requests, batching
  several questions per task, and broad hardware probes only where useful. CPU
  analyses run directly on the lightwork allocation. Later, the user requested
  fewer subagents.
- The user authorized continued execution without repeated approval except for
  genuine scientific course changes, such as replacing ContextCite or changing
  the primary hypothesis. This historical authorization does not override the
  consolidated repository's current task/handoff rules.

## Current scientific conclusions

- The local literal CTP is a poor reconstruction for quality and over-prunes.
- Aggregate-logit CTP is the fairest primary reconstruction currently available,
  but remains an author-unconfirmed best-fit method.
- There is evidence that token identity matters: on the completed holdout,
  aggregate attention's preliminary point estimate is `+1.33` F1 over matched
  global random. Formal canonical inference is still pending.
- There is not yet same-holdout evidence that applying aggregate CTP is better
  than applying no CTP. The older 245-question point estimate was `-1.2163` F1
  versus BTP+QTP, whereas the paper reports a positive CTP-stage contribution.
  That discrepancy remains central.
- The corrected all-kept generation path is a strong internal control. Exact
  reproduction of the authors' unpublished CTP is not established and should
  not be claimed.

## Next steps

1. Reconcile the three historical experiment trackers and relevant commits into
   the consolidated repository under current authority. Preserve branch copies
   and do not overwrite scratch artifacts.
2. Authenticate the exact union of original QIDs 0-35 and batched QIDs 36-1212:
   require all 1,213 QIDs, exactly 21 rows per QID, the frozen hashes/pages, no
   duplicates or gaps, and no global-index/retrieval access.
3. Run the prespecified canonical 100,000-draw nested bootstrap that resamples
   questions and random-mask/support components. Report the 90% TOST against
   +/-1 F1 and the 95% superiority interval; publish a no-replace artifact and
   update the experiment log.
4. Under an explicit active handoff, run BTP+QTP/no-CTP and required native
   controls on the same holdout to determine aggregate CTP's incremental
   quality effect. Do not infer this from the random comparison.
5. Admit and interpret the completed all-visual-removal and regional-attribution
   work before expanding those branches. Keep primary, secondary, and adapted
   diagnostic claims separate.
6. Compare the user's new method against BTP+QTP/no CTP, literal CTP, and
   aggregate-logit CTP under the corrected generation contract. A defensible
   claim is limited to outperforming local reconstructions under matched
   controls unless author code becomes available.

## Context gaps and cautions

- The original dated handoff, paper-ambiguity audit, fair-baseline document, and
  three experiment trackers were available during the chat but absent from the
  consolidated checkout when inspected. Statements above derive from preserved
  conversation and compacted state until their artifacts are reauthenticated.
- Exact numeric results for the visual-softmax normalization-only test and the
  final lightweight quality-only merger publication were not retained in the
  available compacted context.
- Complete repository URLs and full hashes for some paper-code pins, holdout
  identities, and Task 6 artifacts were truncated in the preserved state. Use
  surviving manifests/plans rather than inventing missing characters.
- The exact contents of the user-provided `pasted-text.txt` review were not
  retained, although its approved plan-level corrections are captured above.
- Task 7, Task 8, and later Task 9 outputs require artifact-level admission;
  completion of a Slurm job alone is not a scientific result.
- No implementation or result for the user's own new CTP method was produced in
  this conversation.
- All holdout values labeled preliminary remain historical conversational
  evidence, not canonical published results, until validation and nested
  bootstrap are complete.
