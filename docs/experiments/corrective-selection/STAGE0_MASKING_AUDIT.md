# Stage 0: Col-style guidance for economical teacher masking

Follow-up: [2026-09-13 preparation after the Pro review](STAGE0_MASKING_PREPARATION.md)
supersedes the prospective recipe below with a geometry-checked, structurally
matched R/A comparison. The retrospective results below remain unchanged.

Date: 2026-09-11. Status: retrospective investigation complete; prospective
sampler comparison proposed, not measured. This does not activate a SOL job or
change the canonical [experiment plan](../../ExperimentPlan.md).

## Decision from this audit

**There is enough evidence to test a cheap automatic Col-guided sampler against
random masking. There is not yet evidence that it finds better masks with fewer
reader measurements.** The practical next comparison is a fixed, small bank with
random exploration, soft guidance in both directions, and a few local exchanges.
Compare G-only, pure contrast and gold-aware contrast on each identical measured
bank. No per-question word selection, visual annotation, generated query rewrite,
or hand-picked distractor is required by that proposed sampler.

Ease of execution, repeatability and scale are requirements, not later
optimizations. Prefer cached features, original questions, fixed seeds, explicit
costs and one reusable measurement ledger. The visual review here diagnoses the
sampler; it is not a step in scaled data collection.

## Evidence and scope

- Frozen local receipt: `outputs/colfeatures17-received-2026-09-11/`.
- 17 curated development questions, 68 cached ordered pages, 1,659 region actions:
  725 semantic regions and 934 residual regions. Each case has 10,032 original
  answerer visual tokens.
- Newly received retrieval features are **ColQwen2.5-v0.2**, not the original
  ColPali checkpoint. Each page has 744 image vectors on a 31-by-24 grid, plus
  non-image vectors; projected width is 128. Only scalar historical ColPali page
  scores are available for the older retriever. Cross-checkpoint scalar score
  magnitudes are not comparable.
- Historical Qwen2.5-VL-7B measurements: 256 nonanchor Bernoulli(0.5) region masks
  per question at each of two boundaries, giving 8,704 mask/boundary observations.
  Decoder-input and dynamic-intermediate observations are kept separate.
- This work reuses those observations; no fresh retrieval, query encoding,
  answerer forward pass or answer decoding was performed. The 18 originally
  correct comparison cases are absent from this receipt, so preservation on that
  cohort remains untested here.
- All 68 page images were visually reviewed. The atlas supplies scores, overlays
  and query-token profiles for every action, plus detailed interpretations for
  selected informative sections. It does not claim independent semantic or
  causal adjudication of all 1,659 actions.

Local outputs: [interactive atlas](../../../outputs/stage0-masking-audit-2026-09-11/repeatable/atlas.html),
[page-by-page review](../../../outputs/stage0-masking-audit-2026-09-11/repeatable/visual-review.md),
[region measurements](../../../outputs/stage0-masking-audit-2026-09-11/repeatable/regions.csv),
[priority coverage](../../../outputs/stage0-masking-audit-2026-09-11/repeatable/priority-coverage.csv),
[teacher comparisons](../../../outputs/stage0-masking-audit-2026-09-11/repeatable/teacher-combinations.csv),
[pair diagnostics](../../../outputs/stage0-masking-audit-2026-09-11/repeatable/pair-diagnostics.json).
Generated files remain ignored; scripts and qualitative notes are versionable.

## Matching equations and query variants

Let `M[t,i] = max_j q[t]^T d[j]` over Col image patches overlapping region `i`.
The original late-interaction score sums per-query-vector maxima. See
[ColPali §4.1](https://arxiv.org/html/2407.01449v6) and the pinned
[ColPali-engine processor implementation](https://github.com/illuin-tech/colpali/blob/v0.3.12/colpali_engine/utils/processing_utils.py).
The exported processor uses `Query:` plus **ten** end-of-text augmentation
positions. The original paper describes a different augmentation setup; the
actual exported token list governs this audit. The checkpoint is documented in
the [ColQwen2.5-v0.2 model card](https://huggingface.co/vidore/colqwen2.5-v0.2).

| Diagnostic | Definition | Scalable role / limitation |
|---|---|---|
| All-token region MaxSim | Sum `M[t,i]` over every exported query vector | Original aggregation control; includes prefix and augmentation contributions. |
| Question-only MaxSim `L[i]` | Mean `M[t,i]` over original question positions | Automatic and cheap; preserves contextual vectors but changes aggregation. |
| Augmentation-only | Mean over the ten augmentation positions | Shows whether expanded-query matching changes nominations. |
| Automatic query peak `P[i]` | Maximum, over question positions, of `(M[t,i] - mean_regions M[t,*]) / sd_regions M[t,*]` | Surfaces a region unusually high for any original token; no chosen keywords. A zero-variance position contributes zero. |
| Automatic mix `A[i]` | Mean of within-question rank scores of `L[i]` and `P[i]`, divided by region count | Simple candidate to test; combines broad matching and token-specific peaks. |
| Hand-focused terms | Mean over manually chosen original-question positions | Diagnostic only. Not an independently encoded query or a scalable default. |
| Top-three / mean-patch | Mean of the top three or all patch similarities for each question token, then average query positions | Cheap aggregation controls; change sensitivity to region extent. |
| Unique matching coverage | Sum over question tokens of the all-region maximum minus the maximum without this region | Nonnegative fixed-bank matching loss; overlapping regions can make it zero. Not a reader deletion effect. |
| Score / token cost | Question-only score divided by original reader-token cost | Tested, not adopted; can promote tiny residual regions. |
| Exclusive-centroid score | Question-only MaxSim using patches whose centers map to the region's original reader-token ownership | Geometry sensitivity check; coarse centers leave 50 regions empty. Not a validated replacement mapping. |

All query variants above reuse **the same contextualized original-query vectors**.
Removing a position from the sum does not remove its influence from other vectors.
No new query wording was encoded. Answer-conditioned queries would introduce
different information and extra cost; they are not part of the proposed default.

## What the automatic priorities recover

For each case, nominate the top `ceil(0.2 × number_of_regions)` actions and measure
the fraction of historical fitted coefficient magnitude inside the nominated set.
Then average case-level fractions. The following values use the **input boundary**.
These are coverage of surrogate mass, not accuracy, precision at finding harmful
regions, percentage of distractors found, or deployed answer recovery.

| Priority | Negative-G mass | Negative-C mass | Positive-S mass | Positive-G mass |
|---|---:|---:|---:|---:|
| Uniform region nomination, 200 draws | 20.3% | 20.7% | 20.8% | 20.8% |
| All-token MaxSim | 39.8% | 51.4% | 57.0% | 56.2% |
| Question-only MaxSim | 41.0% | 56.1% | 63.8% | 59.7% |
| Automatic query peak | 44.9% | 53.6% | 59.7% | 46.5% |
| Automatic mix | 43.7% | 56.2% | 63.0% | 61.7% |
| Hand-focused terms | 33.6% | 40.2% | 43.7% | 44.8% |

The automatic mix is a reasonable small candidate, not a proven winner over
question-only matching: their negative-C coverage is nearly identical. Keep both
components available and let prospective measurement yield decide whether the
extra operation earns its place. Hand-written focus terms are not needed to
obtain these results and perform worse in aggregate here.

Region size explains part of the apparent enrichment. Shuffling score priorities
within semantic/residual kind and token-cost quintile gives negative-C null
coverage of 36.8% for all-token MaxSim and 35.1% for the automatic mix, versus their
observed 51.4% and 56.2%. This is a descriptive 400-shuffle sensitivity analysis;
cost bins are coarse and the cohort is curated.

The result changes with the nomination budget. Under a greedy **20% token-cost
cap**, negative-C mass is 30.3% for all-token MaxSim and 38.9% for the automatic
mix, versus 22.4% for the corresponding randomized baseline. Negative-G mass is
21.9% and 28.1%, versus 24.3% random. The largest Q12 distractor costs 2,394 tokens,
so a 2,006-token nomination cap cannot include it at all. A cap on candidate-region
nomination is not the same as the retained-context budget of a teacher mask.

## Visual examples that affect schedule design

| Case and section | What appears on the page | Consequence for masking |
|---|---|---|
| Q07, rank 0 region 2 versus rank 2 region 24 | 66 species-and-subspecies paragraph versus 49-species lead | Strong competing-count probe. Automatic mix ranks the former sixth among 94 regions. Vary it while retaining useful counting-unit evidence. |
| Q12, rank 3 region 59 versus rank 1 region 22 | Unrelated Adele references containing “Chasing Adele” versus football photograph | Raw score ranks references 37/111; question-only ranks them fifth; query peak ranks them first, driven by “chasing.” Large edits must remain possible. |
| Q02, rank 3 regions 75 and 76 | Snow photograph next to “Canopy cover” caption | Supporting visual content and competing text are adjacent. Hand-focused distant-mountain terms rank the caption 105/150; question-only ranks it 13th. Preserve random/local exploration. |
| Q11, rank 1 region 7 versus rank 2 region 11 | Series years versus requested Championship years for the same driver/car | Competition and year binding matter. Full-question features surface the wrong Series table; hand-focused words miss the useful table. |
| Q13, rank 3 regions 28 and 29 | Paramore→Decode bridge plus Decode→Personal Choice theme table | The bridge is useful but can induce the wrong answer field. Sample combinations across multiple backgrounds. |
| Q04, rank 3 regions 24 and 27 | Screenplay prose and infobox contain both writers | A redundancy/completeness control; negative contrast alone does not establish bad content. |
| Q14, ranks 0 and 3 | Historical table continuation split across pages | Preserve the chance to retain both halves. Local page-only edits can destroy a necessary join. |
| Q01, rank 1 region 24 | Team table containing another driver | Raw score ranks it 41/119 and automatic mix also misses the top fifth. Guidance must remain soft. |
| Q16, rank 1 region 19 | “Most wins” field instead of finishing position | An automatic wrong-field nomination; original-question query peak ranks it second. |
| Q17, rank 1 region 5 | Software type and freeware license both appear | Answer-contract ambiguity, not a clean distractor-discovery success. |

The [visual notes](stage0_visual_notes.json) and atlas include all 17 cases and
all four pages per case, including tables with mixed evidence, redundant context,
wrong entities/years, unrelated navigation and possible visual confusions.

## Teacher combinations on the existing bank

Following plan §7, `G` is the maximum mean continuation log-likelihood among
accepted complete answers, `S` scores the fixed original all-keep answer, and
`C = G - S`. These are not correctness probabilities. The region coefficients
for G, S and C are separate historical Lasso fits, so fitted `βC` need not equal
`βG - βS`; raw measured C is always formed from the two measured channels.

For `g = G(mask)-G(all_keep)` and `c = C(mask)-C(all_keep)`, gold-aware ranking first
prefers `g >= -epsilon`; it then ranks admissible masks by `c`, and inadmissible
masks by `g`. Epsilon values 0, 0.1, 0.25 and 0.5 were audited as diagnostics.
No tolerance was frozen here.

At epsilon 0.1 on all 256 input-boundary masks:

- Pure-C selection drops G by more than 0.1 in **5/17 cases**: Q04, Q08, Q09, Q10
  and Q17. Gold-aware and G-only choose different masks in **15/17 cases**.
- **938/4,352** measured masks improve C while lowering G relative to all-keep.
  This is a measured channel tradeoff, not a count of decoded answer failures.
- Restricting to within 5% of each historical comparison budget leaves only
  **352/4,352 masks**, and only **two** match their budget exactly. There are just
  2–52 nearby masks per case. Q07 has zero admissible masks among its four nearby
  candidates at this epsilon. The retrospective bank cannot establish an equal-
  budget winner among prospective samplers.
- At the dynamic intermediate boundary the all-bank pure-C G-drop cases are
  Q04, Q09, Q10 and Q17; gold-aware differs from G-only in 14/17 cases. This is a
  separate teacher surface, not an additional replicate of input deletion.

For a prospective schedule comparison, compute all three teacher orderings on
the same measured masks. This does not require three independently acquired
banks. However, each accepted-G continuation and the S continuation has cost;
sharing a mask or cached prefill does not make continuation evaluation free.

## Historical interaction evidence

Selected pairs were grouped into four binary retention cells in the existing
Bernoulli bank. Other regions vary across rows. These are **conditional cell-mean
differences over varying backgrounds**, not fixed-background factorial replays.
Pairs were selected during this exploratory audit.

| Case | A retained; candidate B added in the grouped comparison | ΔG | ΔS | ΔC |
|---|---|---:|---:|---:|
| Q02 | Snow image; canopy caption | -2.753 | +0.753 | -3.506 |
| Q07 | 49-species lead; 66-count paragraph | -2.342 | +5.216 | -7.558 |
| Q11 | Relevant Championship table; later Series table | -0.292 | +0.609 | -0.901 |
| Q12 | Football image; Adele references | -10.502 | +7.037 | -17.539 |
| Q13 | Theme table; Paramore→Decode bridge | -1.079 | +9.407 | -10.486 |
| Q04 | Screenplay infobox; screenplay prose | -0.070 | -0.043 | -0.027 |
| Q15 | Greyhound highball field; Rickey lead | +0.210 | +0.455 | -0.245 |

Q07 and Q12 also show strong interaction in these averages; Q04 illustrates how
redundancy and shared support complicate the contrast. Selected four-corner
replays can check a few interpretations. They should not become the default
per-question acquisition workflow.

A fixed-alpha five-fold ridge diagnostic on the historical masks has positive
out-of-fold R² for input G on 14/17 cases and input C on 16/17; quality varies
substantially (for example, Q05 has negative R² for both). This is not a validation
of the original full-bank Lasso or generalization to different questions, costs
or guided-mask distributions. See the saved `surrogate-cv.csv`.

## Geometry finding to resolve before strong targeting

A Col patch overlaps a bounding box without necessarily belonging to that
region's original reader-token mask. In the center-ownership diagnostic, median
box-patch ownership is about 76% for semantic regions and 24% for residual regions;
739/934 residual regions have ownership below one half. Residual boxes can inherit
matches from semantic content that their original action does not delete.

The exclusive-centroid alternative improves some nomination metrics, but leaves
50 small regions without a patch and can miss useful captions. Do not silently
replace the action map with it. The next implementation should cache explicit
patch-to-owned-token overlap fractions, preserve the original action identities,
and report empty/poorly aligned actions. This is a feature-mapping correction to
validate, not evidence that the region actions themselves were changed in this
audit.

## Concrete next masking comparison — proposed

Keep the next experiment small. Test the following **three 32-mask banks** at
the decoder-input boundary first, with one common feasible token cost per case
chosen from region costs independently of G/S. Share the same eight agnostic
contexts among banks. Use the same seed sequence and edit-size choices for the
two guided banks.

| Bank | 32-mask composition | Purpose |
|---|---|---|
| R: agnostic | 32 diverse randomized feasible contexts | Measures what guidance must beat. |
| F: full-score guidance | 8 shared agnostic + 8 softly high-score-retaining + 8 softly high-score-removing + 8 local variants | Isolates whether original all-token MaxSim helps acquisition. |
| A: automatic-profile guidance | Same composition, using automatic mix | Tests the cheap original-question alternative without manual focus words. |

The eight local variants should be four pairs based on four different shared
contexts, with two small and two larger exchange pairs. Choose candidate actions
from soft priority plus uniform exploration before reading teacher outcomes.
Use a common cost allocator for all banks; any compensating changes are part of
the measured bundle exchange. Do not call such changes isolated region effects.

One reproducible implementation is cost-conditioned Bernoulli sampling with
region log-weights `0` for R and `±lambda × centered_priority` for the two F/A
directions. A dynamic-programming partition table enforces the common achievable
cost; local variants condition on a fixed outside background. Lambda, edit-size
budgets and seeds must be fixed in the run configuration, not chosen per case.
This is a proposed implementation, not code or masks executed by this audit.

Before scoring, record duplicate masks, actual costs, token-weighted distances,
how often each action is retained/removed, and perfectly co-toggled groups. A
priority is useful only if it creates observable variation; always grouping the
same high-score regions would defeat that purpose. Infeasible local exchanges
must be reported and the bank must not silently acquire extra measurements.

Maximum acquisition for this three-bank proposal is **80 distinct masks per
question** before cross-bank duplicate reuse (32 + 24 + 24), not 96 or three times
that number for the three teacher rules. On 17 cases this is at most 1,360 new
mask contexts, plus any required reference measurement. G alternatives, S
continuations, decoded checks, retries and any later second boundary all count
separately in compute accounting. If that pilot cost is too high, start with R
versus A: at most 56 distinct masks per question. These are prospective counts,
not completed reader work.

Judge the sampler by admissible-mask yield, diversity of informative measured
comparisons, best measured gold-aware utility at the common budget, gold damage,
and small common decoded-answer checks per reader-second. Nomination coverage
alone is insufficient. Compare identical candidate pools when auditing teacher
ordering against complete answers. Repeat the winning simple rule on new
document-disjoint questions before claiming scalability or generalization.

## Reproduction and verification

Run from the DocPrune root with Python, NumPy and Pillow (Pillow must support
`ImageFont.load_default(size=...)`). No model download or GPU is needed.
The output directory must be new. The environment used here was
`/Users/god/miniforge3/envs/mineru34/bin/python`.

```sh
python scripts/stage0_masking_audit.py \
  --root outputs/colfeatures17-received-2026-09-11 \
  --output outputs/stage0-masking-audit-2026-09-11/reproduction
python scripts/stage0_masking_diagnostics.py \
  --root outputs/colfeatures17-received-2026-09-11 \
  --audit outputs/stage0-masking-audit-2026-09-11/reproduction
python scripts/stage0_masking_atlas.py \
  --root outputs/colfeatures17-received-2026-09-11 \
  --audit outputs/stage0-masking-audit-2026-09-11/reproduction
```

The audit verifies source ordering, 10,032-token cost sums, mask shapes and raw
G/S correspondence. The atlas builder verifies every image path and annotated
region identity. Preserve the complete local receipt alongside the generated
atlas: the HTML links to the original page images rather than duplicating them.

Verification completed: eight numerical/data exports reproduced byte for byte
on a second full run; all 34 raw banks and all 1,659 scalar/profile identities
passed direct checks. Two focused numerical tests passed, including held-out
linear recovery and integer/float consistency. Browser checks verified image
loading, question switching, score sorting, region selection and boundary
switching. The visual annotations remain qualitative research judgments.
