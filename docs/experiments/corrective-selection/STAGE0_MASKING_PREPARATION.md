# Stage 0 preparation after the Pro review

2026-09-13. This follows the owner's request to continue and read the supplied
Pro review before execution. It supersedes the prospective R/F/A recipe in
[the retrospective report](STAGE0_MASKING_AUDIT.md). Its historical measurements
remain unchanged. No new reader likelihoods or decoded answers have been produced.

## Changes adopted

1. Match retrieval features to the actual original reader-token ownership before
   using them for guidance. Preserve every action ID and its token partition.
2. Make R the zero-guidance version of the same global/local recipe as A.
3. Compare R versus A first. The all-token F arm remains optional. A beating R
   would not establish that the automatic mix beats question-only matching.
4. Use a common rank-to-logit scale, fixed seed and common budget; report region
   counts because equal token cost does not fix the number of retained regions.
5. Report admissible alternatives and strict preference disagreements, both over
   all pairs and over an explicitly identified comparison set.
6. Freeze an interleaved measurement order, prefix checks and a common decoded
   candidate-selection procedure before observing outcomes.

These changes refine the measurement experiment. They do not change selector
architecture or the canonical ExperimentPlan.md.

## Owned-area mapping: completed

For each cached Col patch, compute its fractional area overlap with the union of
normalized reader-token cells belonging to each original action. Cache all
fractions. Each patch's fractions sum to one; each action's total assigned area
matches its original token footprint. All 68 pages pass those checks.

The initial candidate keeps ordinary MaxSim and only changes patch membership:
use patches with positive-area intersection with the owned footprint, rather
than all patches inside the enclosing box. A `1e-12` threshold excludes numerical
zero. The fractional weights remain available for inspection; they are not
silently multiplied into similarity scores. No model re-encoding is needed.

All **1,659 actions have nonempty support**. The earlier centroid diagnostic left
50 empty. Geometric ownership still does not unmix contextual information within
a Col vector or undo information mixing by the reader's vision encoder.

Input-boundary historical coefficient-mass coverage in the globally nominated
top fifth of actions, averaged over the 17 cases:

| Mapping / priority | Negative G | Negative C | Positive G | Positive S |
|---|---:|---:|---:|---:|
| Original box / question-only | 41.0% | 56.1% | 59.7% | 63.8% |
| Owned support / question-only | 51.4% | 64.2% | 62.5% | 71.0% |
| Original box / automatic mix | 43.7% | 56.2% | 61.7% | 63.0% |
| Owned support / automatic mix | 52.1% | 61.0% | 61.4% | 65.8% |

These remain retrospective surrogate diagnostics. The improvement is not uniform
within action kinds: when separately nominating the top fifth of semantic actions,
automatic-mix negative-C coverage moves from 72.4% to 70.6%; for residual actions
it moves from 22.1% to 26.7%. Global coverage can improve by reallocating nominations
between kinds. The full scope-separated table is saved as `coverage.csv`.

Q01's competing-driver table moves from automatic-mix rank 49 to 27 (question-only
owned-support rank 17). Q07's competing-count paragraph moves from rank 6 to 2.
Q12's Adele references remain first. These examples are diagnostics, not a
selection rule based on hand-annotated roles.

## Matched banks: generated, not scored

The preparation uses the existing 17 questions and 10,032-token original contexts.
The requested retained fraction is **50%**, a disclosed pilot setting independent
of G/S. The largest achievable cost at or below that cap is **5,016 tokens for
every case**. Every generated mask has exactly that cost.

| Component | R | A |
|---|---|---|
| Eight shared contexts | Identical zero-guidance draws | Same draws |
| Sixteen additional contexts | Zero log-weights | Eight positive and eight negative guided draws |
| Eight local contexts | Agnostic edit-pool nomination | Soft priority plus uniform edit-pool nomination |

Each of four shared base contexts generates **two distinct new local masks**.
Outside the nominated action pool, the base is unchanged. Inside it, an exact-cost
conditional draw preserves the base's retained token count. Both arms use the
same pool-packing algorithm and pool-cost-cap sequence: 15%, 50%, 15%, 50% of the
original token population. These are maximum eligible pool sizes, not claims
that every realized edit changes that many tokens. Actual changes are recorded.

The global subset distribution is proportional to `exp(theta @ z)` conditional
on the common exact token cost. A dynamic-programming partition table computes
the distribution; it is not a reader-utility optimizer. R uses theta zero. A uses
the owned-support automatic mix, transformed as
`theta[i] = 1.0 * (2 * (midrank[i] + 0.5) / n - 1)` and its negative. Ties use
midranks. Lambda is fixed at 1, with no per-question outcome tuning.

Local-pool nomination mixes 50% uniform probability with 50% softmax of these
logits; R's weights are uniform. Within a selected pool, both arms draw uniformly
among feasible subsets. Duplicate proposal rejection and pool resampling have
fixed 128-attempt caps and are logged. These are CPU proposals, not additional
reader measurements. Failure to produce the requested bank stops generation.

Mean-centering weights changes the relative probabilities of unequal-region-count
subsets. It is explicitly part of this proposal contract. In the generated banks,
mean retained counts are 48.92 regions for R and 49.02 for A, with respective
semantic/residual means 21.46/27.46 and 21.57/27.44. These observed similarities
are not a general invariance claim.

There are **1,088 arm/slot entries and 945 distinct masks** after reuse. Every
bank contains 32 distinct masks. Every original action is both retained and
removed within each full bank. One R bank, Q03, co-toggles actions 21 and 23;
this is recorded because that bank cannot separate their individual contributions.
No A bank has a nonconstant co-toggled group. Local token symmetric differences
range from 28 to 3,922 for R and 28 to 3,988 for A.

Each eight-slot block contains two shared, two positive, two negative and two
local contexts. Prefixes of 8, 16 and 32 slots therefore have the same family
proportions. R/A are interleaved for physical execution, reusing duplicate masks,
with first-arm order alternating by question number. Outcomes cannot reorder it.

Each block also defines six comparison pairs: base versus each local variant,
local versus local, two positive-versus-negative pairs and the shared-base pair.
That gives 24 predetermined comparison pairs per bank, exported separately from
all 496 unordered pairs. These are **planned comparisons**, not evidence that a
selector has been trained on them.

## Preference-yield audit: implemented and checked on old measurements

Under the current gold-aware rule, S can reverse a strict G-only ordering only
between two admissible masks. With `a` admissible masks, the all-pairs upper bound
is `a*(a-1)/2`. The implementation separately records G ties that C resolves and
strict reversals, so tie-breaking is not counted as strict disagreement.

On the old input-boundary banks at epsilon 0.1 and tie margin 0.01 mean log-likelihood
units:

- All 256 masks: all 17 questions have at least two admissible masks. There are
  **34,887 strict disagreements among 554,880 pairs**, about 6.3%; 15 questions
  contain at least one strict disagreement.
- Within 5% of each historical comparison budget: 15 questions have at least two
  admissible masks, but only 10 have a strict disagreement. There are **193 among
  4,931 pairs**, about 3.9%. Q07 has zero admissible masks; Q09 has one.

These historical pools have varying costs and are **not the newly generated
5,016-token banks**. No preference yield is yet known for the new banks. The
diagnostic grids are epsilon {0, 0.1, 0.25, 0.5} and tie margin {0, 0.01, 0.05};
none was selected to maximize disagreement. Gold-aware winners are reported as
admissibility, g and c, without averaging a numerical encoding of their order.

## Decoded checks and reader execution

The predefined decoded pool is the union of distinct G-only, pure-C and
gold-aware-at-0.1 winners from both banks, plus shared slot 0, R slot 1 and A slot 2
as predetermined controls. This requires at most nine unique masked generations
per question. Generate once per mask and reuse the answer. Hide bank, family and
teacher labels from the adjudication packet, preserving a separate lookup ledger.
The provisional epsilon 0.1 is an audit setting, not a validated tolerance.

The receipt's `input/sources/runtime/run-config.h200.json` records the actual
reader override as Qwen/Qwen2.5-VL-7B-Instruct, revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, Transformers 4.49.0 and SDPA. Its older
top-level Qwen2-VL fields and the geometry-only metadata are not the execution
override. The original prompt, processor contract, tokenization, physical input
deletion and decoding must be reproduced and checked before new scoring.

`src/docprune/correction_depth.py::run_comparison` assumes a 256-mask fitting bank;
it must not be used unchanged for these 32-slot banks. The retained lower-level
forced-intervention scorer can reuse full vision computation and score G/S
continuations, but a bounded adapter, all-keep parity smoke and new SOL handoff
remain to be completed. The old completed Colfeatures17 handoff is not reactivated.

The prospective likelihood budget is 945 new mask contexts plus reference checks;
all accepted-G and fixed-S continuations, decoded checks, setup and retries must
be counted separately. There has been no reader run, model download, GPU
allocation, selector training or change to current execution authority in this
preparation.

## Artifacts and verification

Generated directory: `outputs/stage0-masking-preparation-2026-09-13/`.

- `owned-overlap/`: 68 fractional-overlap caches, case profiles and provenance,
  `regions.csv`, `coverage.csv`, summary.
- `banks-v1/`: current frozen configuration and 17 case ledgers, exact source
  masks, original token-mask hashes, fixed execution order, planned comparisons,
  nomination attempts and bank diagnostics. `banks/` is the earlier identical
  mask set before the decoded-review procedure was added to its config metadata.
- `preference-yield/`: retrospective yield grid and explicit scope statement.

Seven focused tests passed: fractional ownership with small actions, exact
partition values against enumeration, sampled probabilities, attainable budgets,
cost-vector shift invariance, the effect of ordinary centering, and admissibility
constraints on preference reversals (alongside the earlier audit tests). Direct
checks validated all 1,088 arm/slot entries, shared contexts, balanced prefixes,
fixed local backgrounds and cost caps. Q07 regenerated identically. Generating
all 17 banks from the cached profiles took under one second in the observed local
run; this excludes feature extraction and all reader work.

Run with the existing NumPy environment; no new package is required:

```sh
python scripts/stage0_owned_overlap.py --root INPUT_RECEIPT --audit EARLIER_AUDIT --output NEW_OWNED_DIRECTORY
python scripts/stage0_mask_banks.py --root INPUT_RECEIPT --owned NEW_OWNED_DIRECTORY --output NEW_BANK_DIRECTORY
python scripts/stage0_preference_yield.py --root INPUT_RECEIPT --audit EARLIER_AUDIT --output NEW_YIELD_DIRECTORY
python -m unittest discover -s tests -p 'test_stage0_mask*.py'
```

Output directories must be new; inputs are never overwritten. Proposal ledgers
and configuration hashes provide the identities needed by the eventual resumable
scorer. A resumable scorer is not implemented by these generation scripts.
