# DocPrune Random Controls, Retention, and Experiment Program Chat Summary

This is a historical conversation summary, not active execution authority. It
captures the extended DocPrune investigation conducted mainly from 2026-08-24
through 2026-08-28, including compacted conversation state. It overlaps the
earlier CTP/fair-baseline summary but continues through the approved
random-pruning, fixed-retention, throughput, implementation-plan, and cluster
discussions.

At summary time (2026-09-05), the original working checkout
`/home/lmalveau/DocPrune-fix-evaluate-measurement` was no longer present. The
active checkout `/home/lmalveau/DocPrune` described an earlier reproduction
stage in `agent-context/CURRENT_TASK.md`. The results below therefore remain
useful historical evidence, but a future agent must reconcile them with current
authority and authenticate surviving scratch artifacts before continuing.

## Key takeaways

1. **A fair internal baseline was achieved, but exact author CTP was not.** The
   manual DocPrune decoder was corrected to match stock all-kept generation,
   including the complete EOS set and processor inputs. This makes controlled
   comparisons among local CTP methods defensible when all other components are
   frozen. It does not prove identity with the unpublished author
   implementation.
2. **Literal/current CTP caused the clearest controlled loss.** On a sealed
   245-question fixed-page diagnostic, BTP+QTP scored F1 `43.6327`; adding
   literal/current CTP reduced it to `41.3347`, a paired change of `-2.2980`
   with bootstrap interval `[-4.2980, -0.6286]`.
3. **Aggregate-logit CTP is the best-supported local reconstruction, not a
   confirmed paper implementation.** On the same 245 questions it scored F1
   `42.4163`, `+1.0816` over literal CTP but `-1.2163` below BTP+QTP. It retained
   `67.2929%` of post-QTP tokens versus literal CTP's `41.3774%` and closely
   approximated the paper-derived top-4 retention fingerprint.
4. **The initial aggregate-versus-literal quality difference was confounded by
   retention budget.** The 245-question native-threshold comparison changed
   both token ranking and retention. In the corrected 64-question comparison at
   exactly matched per-question token counts, aggregate top-M exceeded literal
   top-M by only about `0.11` F1. The earlier `+1.08` cannot be assigned to the
   ranking formula alone.
5. **Attention did not convincingly beat random pruning on the 64-question
   development cohort.** Aggregate-logit CTP scored `39.921875` F1; matched
   global-uniform random scored `39.427344`, a point difference of `+0.494531`.
   The 95% superiority interval was `[-1.9008, 3.2469]`, so the result was
   officially unresolved. Coverage-matched random scored `39.582031`, only
   `0.339844` below aggregate.
6. **The `20` repetitions applied only to stochastic random masks.** Attention
   policies were deterministic and ran once per question. Each random policy
   used 20 deterministic seeded masks per question, or `64 x 20 = 1,280`
   generations. This was a prespecified local variance-control extension, not a
   claim about what the referenced papers did. Wang's released procedure used
   uniform sampling without replacement, preserved token order, and three full
   evaluation repetitions.
7. **Fixed retention and threshold sensitivity are not the same experiment.**
   Local top-M tests guaranteed exactly 55%, 65%, or 80% retention. The paper's
   `tau_att` sweep retained every token passing a score cutoff, allowing the
   retained fraction to vary by question. Nonetheless, both studies showed a
   similarly flat F1 response over the tested settings; this distinction alone
   does not explain the paper/local quality gap.
8. **Matching retention does not identify the correct attention semantics.** A
   first-generated-token, layer-output, visual-softmax candidate retained
   `65.0360%`, almost exactly the paper-derived `65%`, but produced the same
   answers as BTP+QTP on all 16 diagnostic questions. Its apparent loss to
   current CTP came entirely from one question in a 12-question calibration
   subset and was not strong evidence that the formulas differ in quality.
9. **The broader baseline program was approved and substantially implemented.**
   It compares layer- and budget-matched DocPrune attention, several random
   controls, all-visual removal across forced layer boundaries, and an
   intervention-derived privileged diagnostic. ContextCite with MinerU segments
   is explicitly an adaptation; if impractical, Wang's importance method is the
   fallback.
10. **Use cached pages and features for controlled experiments.** The user made
    this a standing rule after retrieval ambiguity arose: do not run retrieval,
    rebuild the global index, or regenerate cached page features unless
    explicitly asked.

## Research objective and standard of evidence

The immediate research goal is to build a controlled baseline against which a
new CTP method can be compared. Two standards were separated:

- A **usable controlled baseline** requires identical cached pages, BTP/QTP
  masks, model, processor inputs, prompt, decoding, cache/position behavior,
  evaluation, pruning boundary, and token budget, with CTP selection as the only
  changing component.
- An **exact reproduction of author CTP** would require missing paper details or
  author code. The paper omits essential attention extraction, normalization,
  timing, GQA, and cache semantics, so exact identity was not established.

The intended four-arm CTP comparison is:

1. BTP+QTP without CTP.
2. Literal/current prompt-prefill CTP.
3. Prompt-prefill aggregate-logit CTP, the primary best-fit reconstruction.
4. The user's new CTP method.

If future paired results support it, a defensible claim is that the new method
outperforms the literal and best-supported **local reconstructions** under
matched controls. It would not be defensible to claim it beats the authors'
official unpublished implementation.

## Operating constraints established in the conversation

- Preserve all scratch artifacts, failed attempts, runtime worktrees, and
  uncommitted changes.
- Prefer short GPU jobs and multiple GPU sizes for diagnostic coverage.
- CPU-only analysis may run directly on the lightwork allocation without a
  scheduled job.
- Use L40/L40S for canonical results when matching earlier captures. A30,
  A100, H100, or H200 probes were intended for numerical-drift coverage unless
  a protocol explicitly promoted them.
- Do not compare absolute throughput across different GPU types.
- Do not perform retrieval for fixed-page controlled experiments. Reuse the
  already cached ordered top-4 pages and persisted page feature shards.
- The user later requested fewer subagents; future work should be done mainly
  by the primary agent unless delegation is explicitly useful and permitted.
- Continue through nonsequential safe work without repeatedly asking for
  approval; stop only for an input that genuinely changes experimental course.

The old worktree reportedly added the no-retrieval rule to repository and SOL
`AGENTS.md` files. That wording is absent from the active checkout as of this
summary, so the rule should be carried into any future handoff explicitly.

## Paper-versus-code audit

The user objected to an earlier statement implying that everything specified
by DocPrune had been copied. They required a fresh code-base audit, grounded in
the executable code rather than prior chat context, separating guaranteed
paper-faithful behavior from all local choices left unspecified by the paper.
Read-only audit subagents were used at that earlier point, before the later
request to reduce subagent use.

### BTP

Behavior found close to the paper:

- BTP precedes retrieval and QA visual encoding.
- It acts on Qwen/ColPali-compatible 2-by-2 spatial merge groups.
- It converts patches to grayscale, finds a page-wide mode, computes the
  fraction of pixels satisfying strict `abs(pixel - mode) < tau_e`, and retains
  patches with `background_ratio <= tau_bg`.

Unspecified local decisions included grayscale coefficients and arithmetic,
mode tie handling, processor-space patch size, and how four patch decisions are
reduced into a 2-by-2 model token. The implementation used `any` reduction.

Diagnostics:

- BT.601 truncation changed 49 groups and channel mean changed 50, with overall
  retention near `52.2%`.
- Inclusive pixel comparison changed 989 groups but moved retention only from
  `52.2453%` to `51.6292%`, Jaccard about `0.9880`.
- BTP `mean` changed every mask (Jaccard `0.5887`) but was neutral on a
  12-question calibration set.
- BTP `all` retained only `6.93%` and was rejected.

Conclusion: BTP has some unspecified implementation choices but was not the
leading explanation for the controlled quality loss.

### QTP

Behavior found close to the paper:

- It reuses document and question retrieval embeddings.
- It cosine-normalizes and sums similarity across question-token rows.
- It bilinearly resizes from the ColPali retrieval map to the Qwen visual map.
- It applies Gaussian smoothing and retains inclusive `score >= tau_qst`,
  intersected with BTP survivors.

Unspecified choices included prompt/special-token membership, the recovered
32-by-32 map layout, `align_corners=False`, Gaussian `sigma=1.0`, padding and
finite support, 2-by-2 `any` reduction, and sparse/empty-mask behavior.

A direct paper mismatch was noted: the paper/supplement named ColPali-v1, while
the local setup pinned `vidore/colpali-v1.2`. This weakens exact reproduction
claims but does not invalidate internal CTP comparisons when frozen BTP/QTP
masks are shared.

QTP-mean expanded from a promising 12-question screen to a sealed 245-question
test and was not promoted: BTP+QTP F1 `43.6327` versus QTP-mean `42.6653`, delta
`-0.9673`, interval `[-3.1755, +1.0408]`. QTP-mean lowered post-QTP/original
retention from `45.4314%` to `43.4745%`. QTP `all` lost `8.3333` calibration F1.

### CTP

Paper-aligned high-level behavior:

- Select the first layer whose last-token hidden-state L2 norm crosses the
  comprehension threshold, using an inclusive boundary.
- Recompute only the final query against keys at that layer.
- Select visual tokens once, retain passing tokens, keep the selected layer's
  cache full, and compact deeper layers.

Important unresolved choices:

- prompt-prefill token versus generated answer token;
- layer-input versus layer-output comprehension norm;
- raw query-key logits versus post-softmax attention;
- softmax over all keys versus visual keys only;
- mean, sum, maximum, selected heads, weighted heads, or GQA/KV-group handling;
- original-token versus post-BTP/QTP-token scaling;
- cache compaction boundary, generated positions, and M-RoPE handling;
- numerical details of the authors' attention backend.

The literal/current implementation used mean post-softmax attention over heads,
then multiplied visual scores by a visual-token count before applying the paper
threshold. The paper says "attention weights" but does not specify that scaling.

The production factory also reportedly failed to pass configured
`reconstruction_defaults` into `DocPruneQwenAnswerer`; alternate manifests
could therefore describe behavior that never executed. Canonical values
happened to equal defaults, so this did not change the canonical runs.

## Verified baseline results

### Historical full top-4 benchmark

The paper reported all-kept Qwen F1 `36.3` and DocPrune F1 `37.3`, a `+1.0`
improvement. The local historical 2,441-question top-4 results were:

| Condition | F1 | Paired DocPrune change |
|---|---:|---:|
| All-kept | `37.7911` | — |
| Literal DocPrune | `36.7603` | `-1.0307` |

The paired interval was `[-2.1188, +0.0606]`. On single-hop questions, the
change was `-1.6194`, interval `[-3.0910, -0.1609]`; on multi-hop it was
`-0.1531`. These runs predated the EOS correction and are historical planning
evidence, not the final contract for comparing a new method.

The fact that the local all-kept baseline scored above the paper's all-kept
baseline could mean less distractor-removal headroom for CTP. That is a
plausible hypothesis, not a demonstrated explanation. The observed discrepancy
could combine a stronger/cleaner local baseline with literal CTP over-pruning.

### Controlled 245-question stage study

| Stage | F1 |
|---|---:|
| All-kept | `44.8612` |
| BTP | `43.1143` |
| BTP+QTP | `43.6327` |
| BTP+QTP+literal/current CTP | `41.3347` |

The BTP and QTP adjacent changes were not conclusive. Adding literal CTP caused
the supported `-2.2980` loss, interval `[-4.2980, -0.6286]`. A
retrieval-identical full-benchmark stratum of 2,126/2,441 questions still lost
about `-1.1980` F1 with an interval excluding zero, ruling out retrieval-page
differences as the sole cause.

Aggregate-logit CTP on the 245 set:

- F1 `42.4163`.
- Retention `67.2929%`.
- `+1.0816` versus literal CTP, interval `[-0.1102, 2.5796]`.
- `-1.2163` versus no CTP, interval `[-2.8571, 0.1673]`.

Literal CTP retained `41.3774%`; BTP+QTP retained 100% of the post-QTP CTP
population. The aggregate-versus-literal native comparison therefore changed
both ranking and budget.

## Attention-semantics diagnostics and corrections in interpretation

### Aggregate-logit and GQA candidates

The leading prompt-prefill reconstruction averaged raw per-head query-key
logits, applied a visual-only softmax, and multiplied by the post-QTP visual
count. It was selected because it gave a reasonable paper-derived 1/2/4-page
retention fingerprint and improved over literal CTP, not because the paper
specified it.

A Qwen architecture-grounded GQA candidate grouped the seven query heads that
share each of four KV heads, averaged logits within each group, softmaxed each
group map, and then averaged the four maps. Its CPU retention fingerprint mean
error was about `1.72` percentage points versus `2.50` for the earlier
aggregate candidate, and it changed roughly 9% of the selected top-4 mask. It
was worth a bounded diagnostic but remained architecture-sensitive and
paper-unspecified. Its small live QA result did not justify expansion.

Other candidates considered included direct raw-logit thresholds, post-softmax
head sums, maximum heads, alternate visual/full-key softmax domains, and
original-token scaling. Sum of raw logits before one softmax retained only
about `0.1844%` at top-1 threshold `0.5` and was rejected. Maximum-head scoring
fit the paper's count fingerprint poorly. Original-token scaling reached about
`58.01%` retention but hurt calibration quality.

### Generated-token visual-softmax result

The first-generated-answer-token capture used BTP+QTP prefill without prefill
CTP and recorded both selected-layer input and output. Offline analysis found
that first-generated-token, layer-output,
`visual_softmax_after_mean_logits_x_post_qtp` retained `65.0360%` on
calibration versus the paper-derived top-4 target `65.0000%`. This was excellent
mechanistic/count agreement but not quality evidence.

In a 16-question live QA gate:

- On the 12 calibration questions, current prompt CTP scored EM/F1
  `41.6667/46.8333`; the generated-token candidate and BTP+QTP both scored
  `41.6667/43.5000`.
- Candidate versus current had zero wins, 11 ties, and one loss; the mean paired
  difference was `-3.3333` and interval `[-10.0, 0.0]`.
- On four stress questions, all three scored EM/F1 `25.0/25.0`.
- The generated-token candidate's predicted answer was byte-for-byte identical
  to BTP+QTP on all 16 questions.

An earlier response overinterpreted this as evidence that different attention
formulas were not interchangeable. The user correctly challenged that claim.
The revised conclusion was:

> Aggregate-logit, literal, and visual-softmax scoring can select different
> masks, but existing quality evaluations have not established that those mask
> differences reliably change answer quality when budget, timing, and execution
> are controlled.

The apparent `46.83` versus `43.50` advantage came from one question in a tiny
12-question subset. It did not contradict the 245-question result in which
literal CTP was worse, or the corrected 64-question result in which literal CTP
and BTP+QTP were essentially tied. The best current hypothesis is that budget
and timing may matter more than the precise normalization formula, but that has
not been established.

## Corrected all-kept generation contract

The manual decoder initially used only EOS ID `151645`, taken from model config,
while stock Qwen inherited generation-config EOS IDs `[151645, 151643]`. The
manual path could therefore continue after `151643`, creating answer changes
unrelated to pruning.

Historical fixes in the old worktree:

- `a1c0a9ba5d2e5e0e842dc8e15a1db48b05a82162`: shared complete EOS set,
  run-identity recording, and pinned-model mismatch guard.
- `15336753c03a84e34328c5dad521220caa97a18e`: real-model gate under production
  FlashAttention 2.
- `5f350d73d9992f2914f585d843b538528bb7a14d`: exact generated-token parity
  checked before logit tolerance.
- `dd5f000a909a718826541df816a4b65396764e1c`: retained exact suffix parity and
  admitted bounded L40S logit drift with `rtol=0.02, atol=0.07`.

The real processor probe found exact inputs between stock and DocPrune paths:

- `input_ids` `[1, 2537]`, exact;
- exact attention mask;
- `pixel_values` `[10032, 1176]`, maximum absolute difference `0`;
- exact `image_grid_thw`;
- source raster `(1224, 1584)` and smart-resized raster `(1232, 1596)`.

Final admission established exact complete generated-token suffixes on A30,
A100-40GB, H100, and canonical L40S. L40S had reproducible BF16/FlashAttention
first-step-logit drift in 1,710/152,064 values (`1.1%`), maximum absolute
difference `0.06640625`, without output-token divergence. Final L40S job
`62219618` passed with `atol=0.07` in 17 seconds. This is why the controlled
baseline is acceptable even though it is not confirmed author CTP.

## Approved broader experiment program

The user supplied a plan and required full reading of these papers:

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

The plan was critically revised and approved. Important corrections included:

- Random pruning is not an abstract baseline: it must be applied at the same
  forced decoder boundary and to the same post-BTP+QTP visual-token population
  as DocPrune, with the exact same per-question retained count and original
  token order.
- A forced-layer sweep was needed because canonical DocPrune chooses a layer by
  threshold. For diagnostics, each intervention occurs after block `K` has
  executed and populated its full cache but before block `K+1`. This tests token
  utility as a function of depth and makes different policies comparable at the
  same boundary; it is an adaptation, not normal threshold-selected DocPrune.
- All-visual-token removal at the same boundaries tests whether a usable visual
  information horizon exists. Near or beyond such a horizon, selection rankings
  may cease to matter.
- ContextCite is a behavioral attribution diagnostic, not evidence ground
  truth. Using it on intermediate visual states is an explicit adaptation of
  vanilla input-context ContextCite.
- MinerU 2.5 Pro was not used in the cited papers and must be integrated
  independently. MinerU segments page regions; at an intermediate model layer,
  the feasible intervention is a mask over the visual tokens whose spatial
  footprints fall within each segment, not replacement of deep model tokens
  with text segments.
- The proposed privileged score uses the logit of the correct/target answer
  token under interventions, rather than only the token the model happened to
  generate. This remains answer-conditioned and privileged.
- If ContextCite proves too difficult or invalid under these adaptations, use
  Wang's released importance implementation as the fallback rather than forcing
  ContextCite.
- Always obtain and inspect code linked by a paper before synthesizing a local
  implementation. Adapt the published base code only where needed for Qwen,
  dynamic token populations, cache behavior, and the local evaluator.

Three historical project documents were created in the old worktree:

- `docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_PLAN.md`
- `docs/experiments/docprune_random_oracle_horizon_2026-08-26/IMPLEMENTATION_PLAN.md`
- `docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_LOG.md`

Their intended roles were respectively the scientific protocol, the sequential
implementation/execution anchor, and the append-only result log. The relevant
old `AGENTS.md` files were updated to require maintaining them when results or
course changes occur. These files were absent from the active checkout at
summary time and should not be silently recreated without reconciling history.

## Random-pruning implementation and methods

### Native matched-budget protocol

For each question:

1. Run BTP+QTP to obtain the eligible visual-token population `V`.
2. Run aggregate-logit CTP at its native threshold and selected boundary
   `B_l*`; count its retained tokens `M`.
3. Apply every ranking/random comparison at the same boundary to exactly `M`
   tokens from `V`.
4. Preserve original token order in the cache/model execution.

Aggregate native threshold therefore determines the per-question budget. The
literal native-threshold arm keeps its own native threshold and is descriptive,
not matched to aggregate. A separate literal-score top-M arm isolates ranking
at aggregate's `M`.

### Random controls

- **Global uniform random:** sample `M` tokens uniformly without replacement
  from the complete eligible visual population.
- **Page-stratified random:** preserve a page-level allocation before sampling
  identities within pages.
- **Grid-stratified random:** preserve coarse spatial allocation.
- **Coverage-matched random:** divide each page into a normalized 4-by-4 grid,
  count how many tokens aggregate attention retained in each page/cell, then
  randomly choose the same count from that cell.

Coverage matching asks whether aggregate attention selected useful identities
or merely maintained a useful coarse page/spatial distribution. It matches
layer, total budget, per-page allocation, and coarse spatial occupancy while
randomizing token identity. It was a local diagnostic motivated by Wen's
spatial-coverage explanation. Neither DocPrune nor the cited papers were said
to claim this exact procedure, and DocPrune itself was not said to claim it
preserves only semantically important tokens.

### Why 20 random repetitions were used

The run began with 10 masks per question. Global-random Monte Carlo standard
error was `0.5603`, above the prespecified `0.25` target, triggering repetitions
10–19 for all four random controls. This trigger depended on uncertainty, not
which method was winning.

The deterministic attention answer was reused; it was not run 20 times. Random
scores were averaged within question, then compared pairwise with attention.
Nested bootstrap resampling preserved question variation and within-question
mask variation.

This is more extensive than Wang's three full repetitions. It is a local
statistical extension rather than a faithful paper detail. The discussion
corrected the concern that 20 repetitions was an unjustified leap: additional
random draws do not alter the method, and the expansion rule was prespecified,
but reports must distinguish the faithful core sampler from the local precision
extension.

## Corrected 64-question native-M results

The average aggregate-native conditional retention was `66.055800%`.

| Policy | F1 | Notes |
|---|---:|---|
| Aggregate native threshold | `39.921875` | Deterministic; also equal to aggregate score top-M |
| BTP+QTP, no CTP | `39.562500` | No-CTP anchor |
| Literal native threshold | `39.531250` | Own threshold; `40.9280%` retention |
| Literal score top-M | `39.812500` | Same per-question M as aggregate |
| Global uniform random | `39.427344` | 20 masks/question |
| Coverage-matched random | `39.582031` | 20 masks/question |
| Page-stratified random | `38.702344` | 20 masks/question |
| Grid-stratified random | `38.637500` | 20 masks/question |

The attention-versus-global point difference was calculated directly:
`39.921875 - 39.427344 = 0.494531` F1. It was not the midpoint of the confidence
interval. The 95% superiority interval was `[-1.9008, 3.2469]`; its asymmetry
and midpoint near `0.68` arose from the nested bootstrap distribution.

Intuitive interpretation of that interval: the sample is compatible with
aggregate attention being about 1.9 points worse than global random or about
3.25 points better. The observed `+0.49` is encouraging as a preliminary point
estimate, but 64 questions do not establish a real advantage. Aggregate versus
coverage-matched random was only `+0.339844`.

This result was described as motivating but unresolved: random came close to a
specialized attention score under exact layer and budget matching, which
justifies a larger experiment, but it does not prove equivalence or superiority.

## Fixed-retention sensitivity sweep

The fixed-budget sweep kept `round(r * |V|)` tokens at retention rates 55%, 65%,
and 80%. These ranking-only arms used the same forced boundary and same exact
budget at each rate. They were not native threshold policies.

| Retention | Aggregate F1 | Literal F1 |
|---:|---:|---:|
| 55% | `39.890625` | `39.906250` |
| 65% | `39.937500` | `39.796875` |
| 80% | `39.890625` | `39.984375` |

All deterministic EM values were `32.8125`. The ranking curves were nearly
flat, and aggregate versus literal differed by only about `0.11` F1 at matched
native M. This weakened any claim that the exact scoring formula alone caused
the earlier 245-question difference.

Each random fixed-retention policy used only three repetitions per question:

| Retention | Global | Coverage | Grid | Page |
|---:|---:|---:|---:|---:|
| 55% | `37.9115` | `40.6250` | `37.7552` | `38.4583` |
| 65% | `37.4635` | `39.5729` | `40.8333` | `37.5000` |
| 80% | `40.5417` | `39.2917` | `38.1250` | `38.7135` |

These random curves are noisy and nonmonotonic. The 80% global-random result is
not inherently paradoxical: 80% retention means less pruning. However, the
55/65/80 random masks were sampled independently rather than nested; zero of
192 same-question/repetition mask pairs were nested. Changing the retention
setting therefore changed both token count and token identity. These remain
valid matched-budget baselines at each retention, but not a clean causal random
retention curve.

For global random, the three individual repetition F1s at 65% were
`[38.0625, 35.1719, 39.1562]`; at 80% they were
`[39.5625, 41.5781, 40.4844]`. Comparing 80% with 65%, 14/64 questions improved,
48 tied, and two worsened, so a small set of answer flips drove the difference.

### Fixed retention versus the paper's threshold sweep

The paper varied `tau_att` over `0.1`, `0.3`, `0.5`, and `0.7`. Reported F1 was
`31.8`, `31.8`, `32.0`, and `32.0`; decoder throughput was `5.2`, `5.7`, `5.8`,
and `5.9`. The paper did not report the retention percentage at each threshold.

The local sweep specified exact top fractions; the paper kept every token whose
score passed a cutoff. A threshold of `0.5` does not mean 50% retention, and the
same threshold may keep very different proportions across questions. Threshold
pruning therefore preserves adaptive per-question budgets; fixed top-M removes
that adaptivity.

This difference prevents one-to-one settings comparison but is not enough to
explain the quality discrepancy. Both experiments showed essentially unchanged
F1 across their tested settings while throughput improved as more pruning was
induced. The missing paper retention curve and unknown author score scale still
prevent stronger inference.

## Throughput and efficiency observations

The mentor recommended treating latency/throughput as an important secondary
metric and pointed to a simple timing decorator in SelfAug. The existing local
instrumentation used synchronized CUDA events/timing, which is stricter than a
plain `perf_counter` wrapper because asynchronous CUDA work is completed before
measurement.

Throughput means completed work per unit time. The study reported questions per
second separately for encoder, decoder, and end-to-end execution. Exact
generated tokens per second was not saved, so decoder questions/second can be
affected by answer length.

Ad hoc values derived from admitted fixed-sweep raw timings were:

| Retention/policy | F1 | Encoder q/s | Decoder q/s | End-to-end q/s |
|---|---:|---:|---:|---:|
| 55% aggregate | `39.89` | `2.606` | `2.109` | `0.427` |
| 55% literal | `39.91` | `2.605` | `2.077` | `0.422` |
| 55% global random | `37.91` | `2.606` | `2.035` | `0.424` |
| 65% aggregate | `39.94` | `2.600` | `2.006` | `0.422` |
| 65% literal | `39.80` | `2.601` | `1.975` | `0.420` |
| 65% global random | `37.46` | `2.597` | `1.940` | `0.418` |
| 80% aggregate | `39.89` | `2.596` | `1.880` | `0.416` |
| 80% literal | `39.98` | `2.596` | `1.822` | `0.410` |
| 80% global random | `40.54` | `2.594` | `1.868` | `0.414` |

Encoder speed remained stable because CTP occurs after vision encoding. For
aggregate, 55% versus 80% retention improved decoder throughput by about 12%
and end-to-end throughput by about 2.6%. These values had not yet been added to
the experiment log in the compacted state.

Other recorded random decoder/end-to-end q/s values for 55/65/80% were:

- Coverage: `1.965/0.415`, `1.917/0.414`, `1.821/0.411`.
- Grid: `2.019/0.422`, `1.926/0.417`, `1.776/0.409`.
- Page: `1.997/0.417`, `1.907/0.413`, `1.810/0.408`.

Throughput should remain secondary to paired quality, and absolute values
should be compared only on the same hardware/runtime.

## Dataset and cached-feature details

The 64-question fixed cohort used 256 cached pages from 164 unique documents:

- source PDFs: approximately `0.135 GB`;
- feature shards: approximately `1.659 GB`;
- metadata: approximately `0.006 GB`;
- total: approximately `1.80 GB` decimal, or `1.676 GiB`.

The eligible 1,213-question pool used 4,852 pages from 2,081 unique documents:

- PDFs: approximately `1.417 GB`;
- feature shards: approximately `16.871 GB`;
- metadata: approximately `0.0165 GB`;
- total: approximately `18.304 GB` decimal, or `17.047 GiB`.

Cached feature shards are already computed per-document/page visual-token
vectors. They avoid re-encoding pages and ensure every policy receives the same
visual features. They are not additional questions or retrieved-page lists.

A power calculation said approximately 2,199 questions would be needed for the
prespecified ±1 F1 equivalence target at 80% power. The available 1,213-question
pool had estimated power about `0.57998`. It is much better than 64 questions
but may not prove that narrow equivalence. The holdout was not sealed under the
frozen design and required a design amendment before use.

## Last-known task/job status from compacted state

The experiment log was current through Task 6 sensitivity and a Task 7 native
admission-tool checkpoint, but the following completed outputs had not yet been
admitted/interpreted or written into the log:

- Task 7 fixed job `62314816`: all 64 tasks completed `0:0`, exact L40S,
  restart count zero.
- Task 7 native job `62314817`: all 64 tasks completed `0:0`, exact L40S,
  restart count zero.
- Task 9 canonical L40S job `62315446`: completed `0:0` on `scg022`.
- Task 9 A100-80 job `62315546`: last known state pending.

The CPU Task 7 admission tool was built in isolated worktree
`/home/lmalveau/DocPrune-task7-native-cpu-20260828`, commit
`fafaf634e48383179f9e5fc3317f7f9e96d746a7`; 111 Task 7 tests passed. It was
designed to authenticate 128 files and label the result descriptive rather than
claiming an information horizon. A Task 9 admission tool existed at historical
commit `b659` (abbreviated in compacted context), but its completed L40S output
had not been admitted at that checkpoint.

The present filesystem contains later Task 9 worktrees and September logs that
were not inspected for this summary because the user explicitly requested no
new research. Their existence must not be taken as evidence of admitted
scientific results.

## Cluster, VRAM, fair-share, and H200 discussion

The main workload had mostly used L40/L40S GPUs. Recorded peak allocated GPU
memory for canonical full jobs was `18.079996928 GB` decimal, or `16.8383 GiB`,
with a range about `16.08–16.84 GiB`. An earlier conversational estimate of
`28–29 GB` VRAM was corrected. A30 likely fits, subject to a short full-workload
memory gate.

The exact recorded SOL fair-share output was:

```text
Account       User      RawUsage_CHE  RawFairShare  TargetFairShare  RealFairShare
grp_vgupt140  lmalveau  8949.9        0.011086      0.5377527        0.0110860
```

Fair-share was bounded between zero and one and decayed with a seven-day
half-life. Pending jobs do not consume usage; jobs that fail almost immediately
consume only their brief actual allocation. HTC/public/lightwork shared the
same TRES weights in the inspected configuration: CPU `1`, memory `0.25/GiB`,
generic GPU `3`, L40/A30/A100 `25`, H100 `40`, H200 `45`. Thus HTC itself did
not inherently reduce fair-share faster.

Requests for 96 GiB referred to host/system RAM, not VRAM. With memory weight
`0.25/GiB`, 96 GiB contributes weight `24`, approximately the same as one
L40-class GPU's weight `25`. Even short jobs can therefore accumulate
meaningful charged usage if they reserve large host memory. Sharding is valid,
but approximately preserves total work and can add repeated model-startup and
warmup costs; it does not make compute free.

A reconstructed 30-day usage estimate found about `8548.7` decayed CHE versus
current reported `8949.9`. Large contributors included A100 work (`2992`),
A100-40 (`2369`), L40 (`1299`), CPU (`720`), H100 (`635`), and A30 (`487`),
with heavy dates around Aug. 19, 20, 24, and 25. These were explanatory
reconstructions, not a billing-system audit. With no new usage, an approximate
recovery sketch was fair-share `0.011` immediately, `0.105` after 7 days,
`0.324` after 14, `0.569` after 21, and `0.754` after 28; account-wide activity
could change this.

The mentor recommended access to a separate H200 cluster without this
fair-share bottleneck. The user wanted an access message that omitted current
experiment details, SOL progress, and fair-share language while noting the
small dataset and introducing themselves. The preferred message was:

> Hi, I'm Luke Malveaux, one of the new undergraduate students. Eun Woo
> suggested I reach out about getting access to the H200s. He told me that I
> should mention that the dataset I'm working with is relatively small. Would
> it be possible to get access, or could you let me know the process for
> requesting it?

The reply was that storage housekeeping was in progress and access would be
added by the weekend. The likely response was a brief thank-you. Because the
new cluster has H200s and no fair-share issue, large evaluation runs may not
need extensive sharding, although its Slurm partition/account/path details must
still be checked. Slurm (`sbatch`, `squeue`, `sacct`) is common across many HPC
clusters; resource names and local policy are cluster-specific.

## Important conversational corrections

- **"Everything in the paper was copied" was withdrawn.** Core equations were
  implemented, but many required runtime semantics were not specified by the
  paper. Those cannot be called paper-faithful with certainty.
- **Pre-softmax aggregate logits are not confirmed author semantics.** They are
  the best-fit tested reconstruction based on retention and quality evidence.
- **The 16-question generated-token test was not statistically conclusive.** A
  one-question flip made current CTP look 3.33 points better on 12 calibration
  questions.
- **Different masks do not imply different quality.** Existing controlled
  evidence leaves open the possibility that literal, aggregate, and visual
  softmax rankings are effectively interchangeable for answer quality over
  these budgets.
- **The earlier `+1.08` aggregate-versus-literal result did not isolate scoring.**
  Native thresholds retained about 67.29% versus 41.38%; matched-budget results
  were nearly tied.
- **The random `+0.49` comparison was one native-M study, not all random runs.**
  Other fixed-retention random controls used different budgets, masks, and only
  three repetitions; they should not be mixed into that headline.
- **Twenty repeats were random-only and locally chosen.** They improve precision
  but are not presented as a paper-reported protocol.
- **Coverage-matched random is an original diagnostic adaptation.** It tests
  identity within a preserved coarse spatial pattern and is not a claim made by
  DocPrune or a faithful reproduction of Wen code.
- **Higher retention means less pruning.** The temporary confusion about global
  random improving at 80% was resolved, though independent resampling still
  prevents a clean retention-curve interpretation.

## Files and artifacts referenced in this conversation

Historical files from the missing worktree:

- `/home/lmalveau/DocPrune-fix-evaluate-measurement/docs/reproduction/docprune_handoff_2026-08-24/README.md`
- `/home/lmalveau/DocPrune-fix-evaluate-measurement/docs/reproduction/docprune_handoff_2026-08-24/REFERENCE.md`
- `/home/lmalveau/DocPrune-fix-evaluate-measurement/docs/reproduction/PAPER_AMBIGUITY_AUDIT_2026-08-26.md`
- `/home/lmalveau/DocPrune-fix-evaluate-measurement/docs/reproduction/FAIR_CTP_BASELINE_2026-08-26.md`
- `/home/lmalveau/DocPrune-fix-evaluate-measurement/docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_PLAN.md`
- `/home/lmalveau/DocPrune-fix-evaluate-measurement/docs/experiments/docprune_random_oracle_horizon_2026-08-26/IMPLEMENTATION_PLAN.md`
- `/home/lmalveau/DocPrune-fix-evaluate-measurement/docs/experiments/docprune_random_oracle_horizon_2026-08-26/EXPERIMENT_LOG.md`
- `/home/lmalveau/DocPrune-fix-evaluate-measurement/sol/CURRENT_SOL_TASK.md`

Important scratch analyses named in conversation:

- Aggregate-logit 245:
  `/scratch/lmalveau/docprune/ctp-aggregate-logit-stage245-fixed-v1/analysis-final.json`
  (an earlier message also named `analysis.json`; authenticate which is
  canonical before use).
- Generated-token capture:
  `/scratch/lmalveau/docprune/ctp-generated-query-capture-v2/l40s/analysis.json`,
  SHA-256
  `9e371af8177d11640b5635f61be750ce28ac00143bac305d4702e741ac621700`.
- Generated-token live QA:
  `/scratch/lmalveau/docprune/ctp-generated-query-qa-v1/l40s/analysis.json`,
  SHA-256
  `80573b02f2a3b450ad7020eb8c86e1ace30a02179e3db68ddb8a40e5b57dd073`.
- Final L40S all-kept parity root:
  `/scratch/lmalveau/docprune/qwen-all-kept-parity-dd5f000-v4-l40s/result`.

The compacted context did not preserve the exact scratch paths or hashes for
the Task 6 random/fixed-retention analyses. Consult the historical experiment
log or surviving run manifests rather than inventing them.

## Recommended next actions

1. Reconcile the missing historical worktree's commits, plan files, and logs
   with the active `/home/lmalveau/DocPrune` authority. Do not assume later
   worktrees or September outputs are valid continuations without checking.
2. Authenticate and admit the completed Task 7 fixed/native and Task 9 L40S
   outputs before interpreting them. Check the pending A100-80 job or later
   replacement history separately.
3. Update the historical experiment log with admitted Task 7/9 results and the
   already derived throughput table. Label ad hoc or descriptive analyses as
   such.
4. Before a 1,213-question expansion, amend and seal the design, preserve the
   fixed cached page set, and state that power is below the original 80% target.
5. On the H200 cluster, first validate environment, exact model/runtime
   revisions, data paths, processor equality, EOS behavior, and a short memory
   gate. Treat H200 results as a new hardware block unless exact cross-hardware
   output parity is demonstrated.
6. Continue the approved forced-layer/all-visual-removal and privileged-oracle
   program only from authenticated outputs and the canonical plan. If
   ContextCite plus MinerU masks is technically unsound or excessively costly,
   use the prespecified Wang-method fallback.
7. For any new-method comparison, rerun all reference arms under the corrected
   generation contract. Do not compare new corrected runs directly with
   historical pre-EOS-fix scores.

## Unresolved scientific questions

- Does attention selection reliably outperform global or coverage-matched
  random on a substantially larger paired cohort?
- Is there a decoder-layer range in which visual token identity still matters,
  followed by an information horizon where all-visual removal has little effect?
- Does a privileged intervention-derived ranking reveal an oracle gap beyond
  aggregate attention?
- Are literal, aggregate-logit, GQA-aware, and visual-softmax rankings
  effectively interchangeable at matched layer and token budget?
- Does adaptive thresholding provide a quality advantage over fixed top-M
  retention, despite both showing flat sensitivity curves?
- How much of the paper/local difference comes from a stronger local all-kept
  baseline, the ColPali-v1.2 mismatch, CTP scoring/timing, or deeper cache/M-RoPE
  semantics?
- Can any reconstructed CTP produce the paper's positive quality gain under the
  corrected generation contract?

## Context gaps and cautions

- Some original paper/code links gathered into the historical implementation
  plan were not reproduced verbatim in compacted chat context. The arXiv links
  above are known; do not invent repository URLs. Recover them from the plan or
  the papers when that worktree/history is reconciled.
- The exact contents of the user-attached `pasted-text.txt` review and every
  approved change package were not retained in compacted context. The approved
  plan-level consequences are summarized, but the attachment must be recovered
  if wording-level provenance matters.
- Task numbers 1–6 were executed progressively, but compacted context does not
  preserve a complete task-by-task ledger or every commit hash. The last-known
  job status above is more reliable than reconstructing missing steps.
- No result for the user's own new CTP method was reported in this conversation.
- The active checkout contains later Task 9 files dated after this chat's main
  work. They were intentionally not analyzed here, per the instruction not to
  perform new research while summarizing.
- All numerical claims here are historical conversational evidence until their
  referenced artifacts and run identities are reauthenticated.
