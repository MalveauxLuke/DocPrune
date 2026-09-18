# Training Plan

**Status: preliminary design, September 17, 2026. No training, teacher collection, or API calls are launched by this document.** Numerical settings below are starting hypotheses, not measured optima. This specializes the [experiment plan](../../ExperimentPlan.md) for the owner-selected 463-question pilot and incorporates the supplied shared-reference acquisition proposal.

## 1. Recommended first experiment

Start with **64 training questions**, collecting **22 broad masks + 2 fresh random checks + 4 shared-reference single-region flips + 4 follow-ups** per question. Train the native 2B Rich selector with both heads enabled. First freeze the pretrained VLM and train the readout/heads; then compare continued frozen-backbone training with language-backbone LoRA, branching from the same warm-up checkpoint. Use a small, common development set before expanding collection to the rest of the training split.

Begin with the proposed acquisition alone to establish that the complete setup learns. Then, on those same 64 questions, add a cheap random continuation control by reusing the first 24 masks and measuring eight more independent random masks. Do not collect two separate 32-mask banks across all 463 questions up front. Without the random control, describe a learning result, not evidence that adaptive acquisition is better.

**Owner clarification: broad masking is independent Bernoulli(0.5) per region. There is no equal-region-count or equal-token-count requirement.** A mask may retain substantially more or less than half the tokens. Do not repair, reject, pair, or reweight it merely to restore equal retention. Adaptive flips also keep their natural, unequal token costs. The 32-score allowance is a bounded first experiment, not a permanent requirement that all questions receive equal investigation.

The immediate questions are:

1. Do these measured preferences train a transferable Head 1, rather than merely fit the training questions?
2. Does frozen-backbone training suffice initially, or does LoRA improve held-out behavior enough to justify its cost?
3. Does targeted completion teach more useful selection than additional random masks at this small scale?
4. Does Head 2 explain conditional preferences while the directly deployed Head 1 still improves?

## 2. Data, labels, and the early split

### Verified starting point

Use [training463-evidence-filtered-v1](../../../outputs/m3doc600-review-v2/training463-evidence-filtered-v1/README.md). It excludes six confirmed missing-evidence cases and two unresolved cases from the original 471, preserving retained question/answer/page content and order. It is not a completed teacher bank or a new certification that every remaining annotation is perfect. Join all artifacts by QID, never shifted row number.

The retained baseline file currently contains:

| Property | Count |
|---|---:|
| Questions | 463 |
| Normalized exact matches / nonmatches | 191 / 272 |
| Text / table / image questions | 187 / 161 / 115 |
| Original-four / supplemented contexts | 280 / 183 |

Nonmatch does not mean semantically incorrect. Follow [ANSWER_API_VERSION.md](ANSWER_API_VERSION.md): preserve its frozen 471-row adjudication contract, then intersect with the 463 allowlist. Correct cases use S-only preservation; incorrect cases use G/S; uncertain cases remain explicitly ineligible until resolved. Do not silently replace uncertain cases or turn them into errors. The selected API model and completed adjudication receipt must exist before mixed-correctness collection; this plan does not claim they already exist.

### Proposed split: 391 train / 72 development; no test set

Per the owner's clarification, this preliminary study uses training and development only. Development supports tuning, diagnosis and promotion decisions; no separate test evaluation is required for this pilot.

- Freeze QIDs before inspecting new mask results. Stratify approximately by modality, original-four/supplemented context, and adjudicated baseline correctness. Record unfilled strata rather than manufacturing balance.
- Put prior mechanism-audit questions, including the surviving verified ten and the two smoke questions, in training. Identify overlap with the older 17-case audit by QID and normalized question, not by recollection. Record prior exposure for the remaining questions so it remains visible during development analysis.
- Keep exact/near-duplicate question instances together. **Document and page overlap is allowed for this early pilot**, as requested. Record shared document/page counts across splits; do not call the result document-disjoint or evidence of transfer to unseen documents. A later stronger test needs new document families.
- Allocate all 463 QIDs first, retaining uncertain/invalid rows in the split ledger with their exclusion status. Effective train/development counts may therefore be below 391/72. Do not shift development questions into training to fill a target.
- Inside train, select a nested 64-question starter, roughly 26 text / 22 table / 16 image when eligibility permits, covering both correctness and context strata. Use a seeded stratified sample; no OMP-fit, rescue, or mask-score filtering. Aim for at least 16 eligible cases of each correctness stratum if available. Record deviations.
- Inside development, freeze a representative 24-question working subset. Expand evaluation to all 72 for promotion decisions. Both are development evaluations available for inspection and tuning.

Use split seed **1729**, acquisition seed derived from `(version, QID, 1729)`, and initial training seed **0**. Replication seeds **1 and 2** are reserved for the best configuration. Masks and epochs do not increase the number of independent questions.

## 3. Fixed reader measurements and the ranking target

Keep Qwen3-VL-8B-Instruct frozen, with the canonical question/page order, rendering, token ownership, prompt, precision, attention backend, answer limits, and complete-answer contract. Reuse cached ColQwen features and admitted pages; no fresh global retrieval. The supported scoring reference is independent full-prefix scoring. Do not reintroduce the failed shared-KV path or a changed padded-attention backend to make collection faster.

For each eligible question, establish the all-keep anchor under this exact contract. API-version collection must reproduce the adjudicated original answer and capture its actual continuation token IDs. Reusing a historical score or blindly retokenizing answer text is insufficient. A mismatch requires reconciliation, not relaxed string matching.

- **Incorrect baseline:** G is mean log likelihood of the complete correct answer; S is mean log likelihood of the fixed original incorrect continuation; C = G − S. Acceptable alternatives must each represent a complete answer, not separate required list items. S stays fixed across all masks.
- **Correct baseline:** measure S only for the fixed correct continuation. G and C remain null. The selector never receives correctness, answer strings, teacher scores, or OMP coefficients as input features.
- Score differences are in **nats per answer token**, not calibrated probabilities of correctness. A high C can result from destroying both answers; retain the gold-aware rule.

Starting thresholds:

| Setting | First value | Bounded sensitivity check |
|---|---:|---|
| Gold tolerance `epsilon` | 0.10 | 0.05, 0.20 |
| Strict-comparison margin `margin` | 0.05 | 0.02, 0.10 |
| Acquisition channel-change flag `tau_effect` | 0.05 | Initially tied to the comparison margin; do not optimize separately |

These values require a score-scale/yield check on training data and a small numerical-repeat receipt. The observed difference between two attention implementations is not an estimate of stochastic measurement noise. Do not pick epsilon from the baseline-correct/error separation: it measures tolerated *within-question* G loss. For scale only, exp(−0.10) is about 0.905 times the geometric-mean target-token likelihood; that is not “a 9.5% answer-accuracy loss.”

Freeze the initial ordering rule before training:

1. A wrong-case mask is gold-admissible when `G(mask) − G(all_keep) >= −epsilon`.
2. If only one endpoint is admissible, prefer it, unless either endpoint lies within `margin` of the admissibility boundary; then abstain, matching the current implementation.
3. If both are admissible, compare C; if both are inadmissible, compare G. Require a difference greater than `margin`.
4. For correct cases compare S, also requiring a difference greater than `margin`.

Ties are valid measurements without a strict ranking label. Keep all valid comparisons regardless of OMP quality. If a question has no strict pairs, it contributes no strict-ranking loss and remains in population accounting. Invalid targets/runtime/masks are quarantined separately. Conflicting conditional preferences are retained and diagnosed; do not invent a globally consistent Head-1 label by majority vote.

## 4. Phase 1 acquisition: executable decision rules to implement

All region identities and owned reader-token footprints remain stable. Grouping may determine interventions, but this first implementation does not dynamically replace the action vocabulary. A large unsplittable region remains a known limitation; finer ownership would require a new partition/version and new affected measurements.

### Calls 1–22: broad discovery

Independently sample each region bit with keep probability **0.5**. If region i owns `c_i` tokens, retained cost is `sum(z_i*c_i)` and its expectation is half the owned tokens; actual masks need not be close to half. Large regions cause substantial retention variance. Do not “balance” the bits or add compensating deletions.

Cache by full scoring identity and mask bitset. Duplicates reuse their score and do not create duplicate training observations. Draw another mask to obtain the planned distinct bank when possible; tiny action spaces must explicitly stop at exhaustion. Record unsupported zero-visual-token cases as interface limitations instead of silently changing their distribution. All-keep/all-drop behavior must be specified by the interface smoke, not guessed.

Fit centered OMP with an intercept and at most **four active regions**. Use raw G/S channels for incorrect cases and **one S channel** for correct cases. Normalize design columns for fitting; do not variance-whiten response channels or amplify nearly flat S. Record constant/aliased columns and rank deficiency. Select up to four supported regions, filling a short support from the existing absolute association shortlist with seeded tie-breaking. No coefficient becomes a training label.

### Calls 23–24: fresh independent random checks

Sample with the same Bernoulli rule. Save the frozen 22-mask OMP predictions before scoring. Record channel prediction errors, admissibility errors and teacher-ordering disagreements. Two checks can expose a problem; they cannot certify a per-question surrogate. Keep the original nomination fixed for this round. These points become acquisition/training data, not untouched validation.

### Calls 25–28: four flips around one cached reference

Choose b from the 24 measured masks:

- Incorrect cases: among gold-admissible masks choose the one nearest 50% token retention, breaking ties by seeded hash. If none is admissible, choose highest G, then nearest retention, then hash.
- Correct cases: among masks within `margin` of the highest S, choose nearest 50% retention, then hash.

**Near 50% is a preference for a relevant starting background, not an admission constraint.** Do not discard other masks or force b to an exact size.

Score the one-bit variants `b^i` for all four nominees. Each costs at most one new complete-mask measurement because b is already cached. Preserve keep/remove direction, actual token change, ΔG/ΔS, admissibility and teacher preference. If fewer than four distinct mutable regions exist, exhaust the valid tests and report the smaller bank.

Choose a second cached reference b′ by maximum Hamming distance **outside the nominated bits** from b, preferring admissible masks on incorrect cases when available; use seeded ties. If the outside background cannot differ, report that limitation rather than claiming a background test.

### Calls 29–32: bounded adaptive completion

Use this first reproducible policy; do not start with an unvalidated controller-search problem:

1. **One square completion.** Pick the two nominees with the largest observed single-flip channel change `e_i = max(|ΔG_i|, |ΔS_i|)` (S only on correct cases); break ties by nomination order. Score `b^ij`. Together with b, `b^i`, `b^j`, this supplies a two-factor square and two additional single-bit edges. Even quiet first flips can justify this one bounded interaction probe.
2. **Two different-background flips.** Select two distinct nominees, prioritizing an observed OMP direction conflict, then quiet flips (`e_i <= tau_effect`), then larger `e_i`, with nomination-order ties. Score each `b′^i`. Compare oriented *keep minus remove* responses, not raw flip signs, because b and b′ can start in different states. Distinguish raw-channel reversals from gold-threshold-induced preference reversals.
3. **One outside-shortlist probe.** Select an untested region outside the four nominees: a seeded 50/50 choice between the strongest remaining absolute association and uniform sampling of the remaining regions. Flip it around b. If no outside region exists, draw a fresh broad mask.

A cached endpoint supplies information without another reader call. Reassign an unused call to the next eligible action of the same type, then to broad exploration. Record every fallback and do not rescore duplicates just to reach 32 billed calls. Poor OMP checks increase conflict priority; they never invalidate measured teacher preferences. Quiet cases still receive this bounded exploration, then receive lower *future* acquisition priority provisionally.

**Defer automatic three-region cube selection in the first implementation.** Its economics are attractive: b plus three singles need four more corners to complete an eight-setting cube. However, a reliable general trigger is not established. A later preregistered duplicate-region or measured-gating rule can replace the last four calls. No Mozart/Earl-specific rule, manual per-question override, or compulsory third acquisition arm belongs in the first run. Two-factor designs are a standard way to expose conditional effects; the scientific uncertainty here is their training value. See [NIST factorial designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3331.htm).

## 5. Variable-retention banks and what the model learns

**Remove exact-cost validation for this new, explicitly versioned acquisition/training mode.** Keep original exact-cost banks intact for historical reproducibility. All valid measured masks above can teach ordering despite different retained sizes. Also allow recorded G/S-directed acquisition in this arm; it is not the old S-independent supervision-control bank.

For the first pilot, learn a question/context-specific utility function over variable-size subsets. Compute one region-score vector from the full, unmasked context and reuse it for every mask. Do not feed each mask's retained cost back into the encoder/readout: that would change the score vector between a single-flip pair and defeat its concentrated credit assignment.

**Concrete compatibility choice:** hold the existing readout capacity input at **1.0**, representing the full available token capacity, throughout this pilot. All sampled subsets are feasible at that capacity; their scores need not favor retaining everything. Treat deployment retention as an external allocation setting, not a differently conditioned model input. This is a proposed unconditioned-utility pilot, not a claim that budget-conditioned training has already been implemented or tested. A later budget-conditioned experiment can be separately versioned if needed.

Unequal masks intentionally include both content and amount-of-context effects. Record actual tokens and region counts; do not subtract token cost from G/S, area-normalize coefficients, or fabricate equal-cost pairs. Report a trivial “prefer more retained tokens” ranking control and score/retention correlations so success is not mistaken for localization. These diagnostics cost no new reader measurements.

For downstream selection, initially **report a 50% token-cap operating point** using the existing allocator, without imposing that cap on acquisition. The current allocator fills the highest achievable token count under the cap, then optimizes utility; it does not automatically stop when remaining utilities are negative. Keep this behavior explicit. A 25%/75% development sweep or a variable-length stopping rule is a later operating-point choice, not a reason to redo the Bernoulli bank. If the owner chooses another deployment constraint, this training bank remains reusable.

## 6. Selector architecture and loss

Use the implemented **Qwen3-VL-Reranker-2B, question-first, last-layer, Rich readout**. Keep the 2B and 8B native vision encoders separate and frozen. This establishes a reference without resolving cross-model vision sharing. They need not occupy the GPU simultaneously: collect/cache teacher results, unload the reader, then train the selector.

Starting architecture: width **128**, attention heads **4**, region slots **4**, existing two-read Rich structure, geometry/page/token-cost metadata and cached ColQwen query–region features. Pin the actual retrieval schema/dimension and mapping hashes; do not silently disable retrieval because those config fields are currently unset. The selector uses full admitted context; target answers and masked teacher continuations never enter its inputs.

- **Head 1:** `A(z) = sum_i z_i*u_i`. This is the primary deployable scorer.
- **Head 2 enabled:** `A(z) + K(E,z)`. The implemented K is the kept/dropped pooled set correction, not an identified pure interaction model. Its current explicit kept fraction counts regions, not tokens; unequal footprint information also comes through region metadata. Do not describe it as a calibrated G/S predictor.
- Train both with `L = L_rank(A) + lambda2 * L_rank(A+K)`, initially **lambda2 = 1.0**. Use the existing logistic pair loss with temperature **1.0**. Ranking teaches order, not numerical OMP coefficients or measured effect magnitudes. The [RankNet paper](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/) provides the standard pairwise-logistic rationale, not a validation of our teacher target.

Partition measured strict pairs into non-overlapping families, in priority order:

1. **Single-bit edges**, including factorial and different-background edges.
2. **Sparse relative-priority pairs:** two-bit differences between measured variants. When starting bits match, these compare alternative additions/removals; otherwise keep the actual algebra instead of calling it an exchange.
3. **Other broad pairs.**

Average within each nonempty family, then equally across those families, then equally across questions with supervision. Use the same rule in both acquisition arms. Do not let hundreds of broad pairs drown out four isolated tests, nor multiply a question's weight because it has more factorial edges. Record missing families, strict-pair yield and zero-loss questions. Correlated edges remain correlated evidence.

Keep the additive head loss even when Head 2 fits a reversal. If the two heads interfere, inspect Head-1 gradients/dev performance and try **lambda2 = 0.25** on the same bank before changing architecture. A better combined fit without better Head-1 selection supports only a conditional-scorer result. Do not deploy Head 2 reranking for free: it requires candidate sets, scoring time and its own evaluated selection rule.

## 7. Cold start, LoRA branch, and optimization

| Component | Warm-up / frozen branch | LoRA branch |
|---|---|---|
| 8B teacher | Frozen; normally unloaded | Same |
| Native 2B vision encoder | Frozen | Frozen |
| Pretrained 2B language weights | Frozen | Frozen base weights |
| Language LoRA adapters | Disabled or frozen at identity | Trainable |
| Rich readout, feature fusion, Head 1, Head 2 | Trainable | Trainable |

Cold start is **readout/heads training**, not freezing the entire trainable system. First train for **two epochs**, then save a warm-up checkpoint. Branch from exactly that checkpoint into continued frozen-backbone training and LoRA adaptation. Give both an initial **four additional epochs**, with early stopping patience **two development evaluations**. Cap this first comparison at **eight total epochs per branch** unless a documented curve justifies more. LoRA must demonstrate nonzero finite gradients and changed adapter weights; a head-only pass is not an end-to-end adaptation smoke.

The motivation is to let the randomly initialized readout become useful before adapting the pretrained representation. [LP-FT research](https://arxiv.org/abs/2202.10054) supports studying this ordering in vision transfer, but its linear-probe/full-finetuning experiments do not establish an optimum for our nonlinear Rich readout plus LoRA. [LoRA](https://arxiv.org/abs/2106.09685) supplies a parameter-efficient adaptation mechanism, not a guarantee against overfitting 391 questions.

| Hyperparameter | Proposed first setting | Change only when indicated |
|---|---|---|
| Optimizer | AdamW; betas (0.9, 0.999), eps 1e-8 | Diagnose before changing optimizer |
| Warm-up readout/head LR | 3e-4 | Try 1e-4 and 1e-3 on the same cached bank |
| Continued readout/head LR | 1e-4 | Tie to warm-up choice; avoid a full cross-product |
| LoRA LR | 2e-5 | Compare 1e-4 if stable but underlearning |
| Weight decay | 0.01; zero on biases/norms | Try 0.05 only for a clear train/dev gap |
| LoRA rank / alpha / dropout | 8 / 16 / 0.0 | Rank 16 only after an adaptation bottleneck is evident; dropout 0.05 for overfitting |
| LoRA targets | language q_proj, k_proj, v_proj, o_proj | Keep vision and MLP targets out of the first sweep |
| Gradient clip | global norm 1.0 | Log clipping frequency; don't conceal exploding gradients |
| Microbatch | 1 question with all its masks | Larger only after measured throughput benefit |
| Accumulation | 4 questions per optimizer step | Keep effective batch fixed in branch comparisons |
| Precision | BF16 pretrained model; FP32 loss/reductions and current readout parameters | Smoke dtype/device compatibility; no new quantization initially |
| LR schedule | Constant for the short warm-up; 10% linear warm-up then cosine for branch training | Step by optimizer updates, not masks |

One encoder pass supplies all 32 mask scores for a question; **32 masks are not 32 VLM training forwards**. With 64 supervised questions and accumulation four, an epoch has approximately 16 optimizer updates, fewer if questions have no strict pairs. Report actual update counts; an “epoch” is not comparable when the eligible pool changes. Shuffle questions per epoch deterministically. Match branch data order and optimizer-update allowance for the first adaptation comparison; separately report GPU time.

For frozen-backbone training, cache detached last-layer visual/question memories under exact checkpoint/prompt/grid identities, then run the trainable Rich readout on them. Do not cache its final region embeddings while that readout is learning. LoRA changes language representations and therefore invalidates those hidden-state caches; only validated frozen-vision features can remain reusable. Cache sizes must be measured before materializing the full cohort.

Transition checkpoints must explicitly save the warm-up state, identity LoRA configuration, optimizer/RNG provenance and new phase. Start fresh optimizers for **both comparison branches** so inherited momentum is not an extra difference; only LoRA has adapter parameter groups. Do not pretend the transition is a strict resume of the same set of trainable parameters. Follow the installed [PEFT LoRA interface](https://huggingface.co/docs/peft/en/developer_guides/lora) and pinned environment rather than upgrading packages to match current documentation.

## 8. Small random comparison and honest accounting

Proposed arm: **22 shared random + 2 shared fresh random checks + 8 controlled follow-ups**. Random arm: **the same 24 random masks + 8 additional independent Bernoulli masks**. Both receive the same teacher ordering, model initialization, family-normalized loss, training schedule and development evaluation. The random arm's last eight ignore outcomes. It does not receive adaptive-arm masks or OMP predictions as labels.

| Acquisition for 64 training questions | Distinct mask contexts, before cache hits |
|---|---:|
| Proposed arm alone | 64 × 32 = 2,048 |
| Extra to build the random arm | 64 × 8 = 512 |
| Union of both banks | 2,560 |
| Two separately collected banks would have cost | 4,096 |

This adds 25% to the proposed bank's mask count, not 100%. These are **mask-context measurements**, not target-continuation calls: incorrect cases require G and S scoring; correct cases require S only. All-keep anchors, exact-answer reproduction, retries, numerical repeats, fresh dev scores and decoded-answer evaluation are additional work. Actual runtime depends on retained tokens and answer length. Equal numbers of masks do not imply equal compute; show both counts and measured GPU-seconds/token load. There is no requirement to keep retention or future acquisition budgets equal.

Initially compare the acquisition arms with the selected cheap frozen-backbone training recipe. Do not immediately run the full acquisition × adaptation × architecture × seed factorial. If LoRA is necessary to show learning at all, use the same selected LoRA recipe for both. A proposed-only positive result can justify this small comparison; it cannot justify claiming acquisition superiority. A negative proposed-only result calls for label/interface/fit diagnosis before automatically buying more masks.

## 9. Development, tuning, and promotion

Prepare **eight independent Bernoulli masks per working-dev question**, using a separate RNG stream and no adaptive controller: 24 × 8 = 192 contexts, plus anchors. Freeze a common evaluator version and thresholds. Do not tune each training arm against its own acquisition distribution. Compute Head-1 and combined pair accuracy, margin/loss, and channel/admissibility strata on these saved measurements. Track retention-only ranking as a cheap reference. This dev bank is sparse evidence, not a full oracle over region subsets.

Tune sequentially, reusing the same teacher bank:

1. Audit strict-pair yield across the small epsilon/margin sensitivity grid offline. Freeze the first runnable label version; changing it creates a new version. Low yield is information, not permission to shrink margins until almost everything becomes a label.
2. Try up to three warm-up learning rates with seed 0 and short runs. Continue the best plausible rate; include the untrained readout baseline. Reject nonfinite/constant-output failures immediately.
3. Compare frozen continuation with LoRA at 2e-5; try 1e-4 only if the first is stable and insufficiently adapting. Use at most one targeted follow-up, such as smaller Head-2 weight, for a diagnosed issue. **Cap this first search at eight distinct configurations**, excluding seed repetitions.
4. Repeat the selected configuration with seeds 1 and 2 before attributing a small gain to the method. At this scale, variability and per-question differences matter more than a single best number. Small targeted searches avoid a wasteful Cartesian grid; [random-search research](https://www.jmlr.org/papers/v13/bergstra12a.html) motivates bounded search, not a guarantee of finding an optimum.

Select interim checkpoints using the **same fixed development ranking target**. When epsilon/margin are changed for training, do not compare incomparable training losses or redefine the dev target to favor each model. Decode only shortlisted checkpoints, not every epoch of every trial. Because likelihood and answer correctness differ, final promotion depends on decoded behavior too.

On the working-dev subset, reuse the unpruned baseline and generate answers for:

- learned Head-1 selection at the declared retention operating point;
- one deterministic ColQwen-guided regional selection;
- three fixed random regional selections, sampled independently with Bernoulli(0.5).

All arms record actual retained tokens; random masks are not repaired for exact equality. Report individual costs and quality together. This is a practical policy comparison, not proof of equal-token efficiency. Reuse cached generations when the complete inference identity matches. Add no teacher-selected best-of-bank result to the deployable score; that is separately labeled oracle headroom. Full-answer grading follows the frozen evaluator with an explicit uncertain count; the original-answer API label cannot grade a newly generated answer.

Report overall accuracy, **incorrect→correct recovery**, **correct→incorrect damage**, original-four versus supplemented context, modality, actual tokens, reader latency, selector latency and total cost. Use paired per-question comparisons and intervals; acknowledge shared-document dependence and the wide uncertainty of 24/72 development questions. Visual inspection should include successes, damage, quiet cases and OMP failures, not just pleasing saliency maps.

Proceed to a nested **160-question training set**, then at most the eligible **391**, only if interfaces are sound, training yields usable preferences, and development behavior merits the additional collection. Reuse the initial 64 banks and keep the same split. Record method, threshold and operating-point changes alongside development results. These are feasibility and tuning results; an independent final test is outside this pilot's scope.

## 10. Necessary implementation before execution

The repository supplies most components but **does not yet execute this plan unchanged**:

| Existing behavior | Required bounded change |
|---|---|
| `TeacherBank.validate` enforces equal achievable token cost and normally S-independent proposals | Add a versioned variable-retention, adaptive-acquisition bank; preserve identity, validity and historical-mode checks |
| Collector freezes the old proposal bank before teacher scores | Implement and journal the 22+2+4+4 state machine, caches, predictions, reasons and resume cursor |
| Current OMP helper expects 22 rows and two outputs | Support S-only correct cases explicitly; never duplicate S into a fake G channel |
| Strict split verification rejects document/page overlap | Add a named query-disjoint pilot split policy with duplicate-QA protection and overlap receipts; retain strict mode |
| `ProxySelector` constructor requires trainable LoRA | Support frozen-pretrained warm-up and an explicit phase transition, while still verifying LoRA gradients in adaptation |
| Training uses a single learning rate and no described phase/accumulation controls | Add parameter groups, accumulation, schedule and phase-aware save/reload; retain existing norm-1 clipping |
| Loss averages all strict pairs together | Add deterministic pair-family assignment and normalization |
| Readout capacity is derived from the allocation budget | Separate fixed full-capacity utility conditioning from the externally selected deployment cap in this pilot mode |
| Default configs use shared vision and Head 2 off | Explicitly select native vision, Rich, retrieval schema and Head 2 on |
| Native vision currently runs in forward; frozen-memory reuse is not a completed contract | Add only the cache reuse needed by this pilot, validate parity, and invalidate correctly on LoRA activation |
| Workflow does decoded validation per epoch | Permit cached development-ranking checks and bounded decoded promotion checks |

Main code references: [experiment.py](../../../src/docprune/stage2/experiment.py), [qwen.py](../../../src/docprune/stage2/qwen.py), [readouts.py](../../../src/docprune/stage2/readouts.py), [policy.py](../../../src/docprune/stage2/policy.py), [supervision.py](../../../src/docprune/stage2/supervision.py), [training.py](../../../src/docprune/stage2/training.py), [workflow.py](../../../src/docprune/stage2/workflow.py), [collection.py](../../../src/docprune/stage2/collection.py), and [OMP discovery](../../../experiments/omp10/discovery.py). Do not disable validation globally to get a run through.

## 11. Smoke and SOL execution sequence

1. **Local preparation:** complete adjudication prerequisites, freeze split/label/acquisition versions, implement the changes above, and run focused CPU tests of variable sizes, one-bit identity, adaptive resume, S-only routing, family loss and phase checkpoints. No giant test sweep is necessary.
2. **Owner-approved two-question smoke:** choose two of the verified incorrect ten still present in 463; collect 32 masks each, train Rich/Head 1/Head 2, perform a genuine LoRA update, and save/reload. Confirm token deletion, continuation alignment, finite G/S, correct parameter freezing, loss gradients and score parity after reload. A wrong-only smoke does not validate the correct-case S-only interface: add one bounded correct-case check before mixed-stratum collection, without another 32-mask wrong-case repeat.
3. **Measure, then request resources:** use the SOL browser shell and a compute allocation for execution, following the [SOL operational context](../../../agent-context/modules/sol-browser-startup.md) and [fairshare/batching record](../../../agent-context/modules/sol-fairshare-batching.md). Re-read current SOL rules and live scheduler minima at submission. Transfer code through Git and untracked inputs/caches with recorded rsync procedures. No dataset processing or model work on a login node.
4. **Run the 64-question acquisition and short training:** load the teacher once per useful resumable chunk; checkpoint scored questions so wall-time expiration does not discard them. Unload it before selector training. Start with one GPU, one-question microbatches, minimal CPU/RAM satisfying measured needs and scheduler minima, and any compatible GPU. Training memory must be measured; historical inference memory is not a training guarantee.
5. **Choose queue/fairshare tradeoffs from receipts:** request observed peak host/device memory plus a small explicit margin, and wall time covering load + measured work + checkpoint overhead. Compare live queue estimates before unnecessary sharding. GPU utilization, allocated memory, peak memory, GPU-seconds, billed TRES, model-load time and useful work all belong in the receipt. Empty VRAM alone does not justify larger batches; accept occasional bounded timeout/OOM over systematic over-requesting, as directed.

Keep all large caches, masks, checkpoints and outputs ignored; version small configs, manifests and reports. Required ledgers: source/exclusion hashes; split and exposure IDs; adjudication identity; reader/runtime identity; mask bitsets, actual costs and parentage; frozen OMP predictions; full measured G/S; pair labels/families; trainable parameter lists; phase/seed/checkpoint identities; decoded outputs and grading; actual scheduler/resource costs.

## 12. Interpretation and subsequent decisions

- **Clear preferences, useful Head-1 dev gains:** proceed with the small random comparison and learning curve. Visual localization supports interpretation but does not replace downstream evaluation.
- **Clear preferences, poor OMP:** keep the labels; use the random comparison and conditional checks to assess nomination value. Do not throw away the question.
- **Training fit without dev gains:** examine shared-document exposure, sample diversity, head capacity and adaptation before buying more masks per question. More independent questions may be more valuable.
- **Head 2 improves but Head 1 does not:** report the representational limitation; consider separately priced candidate reranking or distillation later, without claiming direct selection succeeded.
- **Few meaningful preferences:** report yield, preserve quiet-case records and avoid forcing labels. Revisit only with an explicit new hypothesis or resolution/background change.
- **Scores improve but decoded answers do not:** retain the distinction between likelihood supervision and the desired QA outcome; do not promote solely on OMP fit or training loss.

The acquisition rationale is informed by the local [ten-question visual audit](../../../agent-context/findings/2026-09-17-omp10-visual-audit.md) and the supplied proposal. Ablation surrogates such as [ContextCite](https://arxiv.org/abs/2409.00729) motivate probing response dependence; they do not establish that OMP-guided follow-ups improve this selector. The first study is intended to find that out with reusable measurements and a small, transparent tuning allowance.

## Execution update: frozen quality-first split

Owner explicitly prioritizes good, already-analyzed questions; prior exposure and
document overlap do not exclude cases in this pilot. Frozen package:
`outputs/training-pilot-v1/split-quality-v1/`. Source463 preserved;391train/72dev.
Train64 has26text/22table/16image and32correct/32incorrect; workingdev24 has
10text/8table/6image and12correct/12incorrect. Includes12ready historical17 cases
and9ready ten-audit cases; both correct smoke questions are in train64. Owner
approved recording five historical cases without current packages as candidates,
and holding the227 missing-header case pending repair. All12uncertain labels stay
quarantined; effective fulltrain378, dev72. No test set and no unseen-document claim.
Selection uses frozen evidence reviews/labels, no mask outcomes. Quality here means
existing evidence support, not guaranteed pruning benefit or another complete visual audit.

Smoke63542785 passed in4m41s, with52.63s reader collection for2questions and35.89s
selector work. It verifies wiring, not improvement: half-context masks reducedS
onbothcorrect cases. Gradients clipped onall10updates; log/inspect64development
curves before changing LR. Reader peak reserved20.06GB (collection); training6.97GB.
Cached/uncached parity exact, nativevision reused16times andhiddenmemories14times.

Next launch is collection only:64x32 training masks plus24x8 fixed independent dev
masks, reusing64completed smoke masks. S-onlycorrect/G+Sincorrect; full-prefix,
canonical baseline regeneration, one reader load, one question resident at a time.
No new ColQwen/MinerU processing. One genericGPU,2CPU,24000M and45min initially,
subject to live queue comparison; retain per-question/per-mask checkpoints if time
expires. This is an A100-informed estimate, not a guarantee on every compatibleGPU.

After bank integrity/yield checks, train both branches from the same warmup with
streamed examples and bounded/disk-backed frozen caches. Evaluate Head1 onthe24dev
questions against untrained/token-count/retrieval/random controls. Cached ranking
metrics do not replace later matched decoded-answer checks. Do not queue the current
two-question-only training runner as a64-question production job.
