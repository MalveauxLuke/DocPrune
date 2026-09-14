# Stage 0 adaptive teacher acquisition, frozen v1

2026-09-14. Owner-authorized scope: freeze the policy, implement and test locally,
and connect the frozen reader. GPU smoke and experimental measurements are the
next stage and have not been run. This specification extends the prepared R/A
comparison in [STAGE0_MASKING_PREPARATION.md](STAGE0_MASKING_PREPARATION.md).

## What the experiment tests

At 32 revealed masked contexts per question, does feedback-directed acquisition
find better admissible contexts or more useful controlled preferences than the
prepared banks? All arms use the same 17 development questions, original action
partition, ordered four pages, frozen Qwen2.5-VL-7B reader, physical deletion at
B_input, and exactly 5,016 of 10,032 original visual tokens.

| Arm | Questions answered by comparison |
| --- | --- |
| R: existing zero-guidance bank | How much can the fixed global/local recipe discover without retrieval weighting? This is the random baseline, not unconstrained independent token masking. |
| A: existing retrieval-guided bank | Does soft retrieval guidance help that same prepared recipe? R and A remain byte-for-byte unchanged. |
| Adaptive: reversible groups and teacher-directed follow-ups | Does the combined acquisition policy improve discovery per measurement over A and R? |

Adaptive versus A changes acquisition and grouping together. A positive result
would not isolate grouping's individual benefit. A separate fourth arm is deferred
until there is evidence that this combined policy is worth dissecting. The 17
questions informed this design; they support development decisions, not a held-out
generalization claim. No selector is trained here.

Teacher changes determine behavioral usefulness. Retrieval only supplies a prior
for initial grouping. Low-priority regions need not all receive fine-grained
measurements, but low retrieval rank alone does not certify behavioral irrelevance.
Original regions remain accessible in every fine exploration draw and audit.

## Teacher contract and ordering

- Targets are the exact saved answer token sequences, without appending EOS.
  G is the maximum mean token log likelihood across the accepted complete answer
  continuations. S is the mean log likelihood of the fixed **actual original
  full-context generated answer**. It is never replaced by a new masked answer
  or a stale reference written in case notes. C = G - S.
- Reference G/S comes only from the authenticated historical full-context
  likelihood vector. Historical masked scores are excluded from the portable
  inputs. At execution the reference is rechecked against the same frozen reader.
- A context is admissible when G >= G_reference - 0.1. Among admissible contexts,
  order by C, then G, then -S. Inadmissible contexts come after all admissible
  ones and are ordered by G, then C, then -S. Exact remaining ties use the
  lexicographically smallest canonical mask hash.
- Raw responsiveness is distinct from desirability. The first eight observations
  freeze scales sG=max(0.01,IQR(G)) and sS=max(0.01,IQR(S)), using NumPy's default
  linear percentile interpolation. For an edge, R=max(|deltaG|/sG,|deltaS|/sS).
  A pair is quiet only if both raw absolute changes are <=0.01. C staying constant
  while G and S both move is responsive.
- The 0.1 guard, 0.01 margins, and scale floor are provisional, pre-frozen pilot
  choices, not estimates of numerical noise or statistical confidence.

G/S supplies acquisition-time supervision when answers are known. This experiment
does not establish that a deployed selector can access gold answers or duplicate
these choices without teacher measurements.

## Reversible grouping

Preserve every original action ID and token owner. Rank actions by descending
owned-support automatic retrieval priority, breaking ties by original index.
Keep the top ceil(0.2*n) as individual actions. Also keep any action whose cost
exceeds floor(0.05*10032)=501 tokens individual.

For remaining actions, bucket by page, source kind, and a 3-by-3 tile of the
mean owned-token grid row/column. Use grid indices directly, without adding 0.5:
tile_row=floor(3*mean_row/H), tile_col=floor(3*mean_col/W). Sort each bucket by
mean row, mean column, then original index. Next-fit pack groups up to 501 tokens.
There is no member-count cap. Sort resulting groups by their smallest member.

Verified on the sealed geometry: 1,659 original actions, 72–150 per question,
become 48–80 initial groups; all 17 grouped subset spaces can attain 5,016 tokens.
These are feasibility results, not evidence that the grouped actions are harmless.

Seed masks can already split a group. A follow-up intersects each group with each
side of the observed parent difference. It never rewrites an observed parent to
make it fit a group. Opening a group is local to that comparison; it does not
permanently label children important or unimportant.

## Exactly 32 observations

The adaptive seed is **A slots 0–7**. These are not eight entirely random masks:
they preserve the initial mixed A recipe. Then repeat three rounds:

1. One broad grouped draw and one broad original-action draw.
2. One focused complementary quartet: up to two new contexts.
3. One fine audit quartet: up to two new contexts.
4. Another focused quartet: up to two new contexts.

This allocates 8 seed + 6 broad + 12 focused + 6 audit slots. There is no early
question stopping, cross-question reallocation, or later epoch in v1. All arms
receive exactly 32 distinct logical contexts per question. Later reallocation
requires results from this fixed-budget test.

Broad draws are uniform over feasible exact-cost subsets of their respective
units, using the log-partition dynamic program from the original bank sampler.
Uniform subsets do not imply equal individual inclusion rates for unequal costs.
Grouped draws are retrieval-informed through grouping, even with zero log weights.

The deterministic random generator is NumPy default_rng, seeded by the first
16 hex digits of canonical SHA256([20260913, case, operation metadata]). Broad
operations include round and granularity; follow-ups include round, phase, mode
and sorted parent hashes. Focus and audits have independent random streams.
All JSON hashes sort keys, use compact separators, retain ASCII escaping and
reject nonfinite values. The production runtime pins NumPy 1.26.4.

## Which comparison receives a follow-up

Maintain a frontier of measured equal-cost context pairs. Initially this includes
all 28 pairs among the eight seeds. Each broad context connects to every already
revealed context. Explicit fallback A contexts also connect this way. A completed
quartet adds its four smaller parent-child edges, rather than pairing each local
child with the entire history.

An edge's first discovery fixes its orientation, depth and inherited discount.
Rediscovery does not reset it. Root depth is zero; children add one. Each edge may
be expanded at most once in focus mode and once in audit mode. Pairs that cannot
be partially recombined even at original-action resolution are ineligible; that
is a property of this pair, not an atomicity claim about its regions.

For d differing original actions, focus priority is:

`discount * R / (sqrt(d) * (1 + depth))`.

Focus considers responsive, not-yet-focused, recombinable edges. If any touch an
admissible endpoint, restrict to those and maximize priority. Otherwise first
maximize the better endpoint's G, then priority. Further ties prefer the better
gold-aware endpoint, smaller d, then sorted endpoint hashes.

Audit includes quiet pairs and always splits at original-action resolution.
Its coverage score is the mean of 1/(1+audit_count_i) over differing actions;
audit counts increase for all actions in the selected parent's difference after
completion. In rounds one and three, minimize R/sqrt(d), then maximize coverage.
In round two, maximize coverage first, then minimize R/sqrt(d). Remaining ties
use smaller d and sorted endpoint hashes. Audits have no retrieval weighting.

## Controlled complementary exchanges

For equal-cost measured parents u and v, let I be actions only in u and J actions
only in v. Both sides cost T. Find a common reachable proper subset cost 0<t<T
nearest T/2, breaking ties toward smaller t. Sample I1 and J1 independently and
uniformly among subsets of their side's current units with exact cost t.

Construct w=u-I1+J1 and x=v-J1+I1. Both remain at 5,016 tokens. Everything outside
I union J stays fixed, and w+x=u+v at the original-action level. These are two
backgrounds for the **same exchange**, not two unrelated masks.

Focus starts with intersected group units. If no proper common cost exists, open
the largest-cost multi-action unit; ties use greater member count, smallest
original index, then side I before J. Repeat until feasible or fully individual.
Audit uses individual units from the outset. The chosen candidate edge is logged.

Proposal draws have a fixed 128-attempt cap. A quartet is accepted when at least
one child has not yet been revealed to this arm. A child already revealed to this
arm is reused without another logical observation. Complete the quartet, then
fill unused measurement slots with the first unseen A mask. No eligible edge,
infeasible grouped broad draw, or duplicate-cap exhaustion also uses this explicit
fallback. Each fallback records its cause. A's 32 distinct masks guarantee an
unseen fallback while the adaptive arm has fewer than 32 observations. Fallback
frequency is a primary implementation diagnostic, never hidden as successful
adaptive investigation.

## What a quartet teaches and when to reduce attention

Record delta(w-u) and delta(v-x) separately for G and S. Their comparison tests the
same swap in two retained backgrounds. Also record interaction Y(u)+Y(v)-Y(w)-Y(x)
for both channels; zero interaction is not a reason to skip a quiet-parent audit.
Cancellation can produce useful opposing children despite flat parents.

A margin-aware direction prefers the higher C when both endpoints are admissible,
or higher G when both are inadmissible, requiring a difference >0.01. If the pair
crosses the admissibility boundary, require both endpoints to be at least 0.01
away from that boundary; otherwise direction is undetermined. Opposite nonzero
swap directions across backgrounds are a reversal.

A follow-up is productive if a child meaningfully beats the pre-follow-up
incumbent under that rule. If both backgrounds give the same nonzero direction
and neither child improves the incumbent, multiply new descendant priorities by
0.5. This reduces effort without deleting the branch or making quiet descendants
unavailable to audits. Stable low-impact areas need not be exhaustively resolved.

These preferences are relative balanced exchanges; removing an action and adding
compensation cannot establish a unique isolated effect for that one action.

## Execution, accounting and outputs

A new runner uses the existing physical-deletion decoder and teacher-forced
likelihood primitives. One frozen full-vision checkpoint is retained per question,
then each online mask resumes from it. No BTP/QTP, fresh retrieval, global index,
new page rendering, target re-tokenization, or historical surrogate is involved.

R/A/adaptive histories are separate. For each slot, odd-numbered questions reveal
R then A then adaptive; even-numbered questions reveal A then R then adaptive.
A shared physical cache is keyed by the run/reader/input identity and original
visual-token mask. The controller receives a cached score only after requesting
that mask itself. A cache hit still consumes one logical observation for that arm.

Before a physical call, atomically publish the exact proposal and revealed-history
hash. Publish a checksummed physical result before its logical result. Resume
replays the deterministic policy and rejects changed identities, proposals or
scores. A single-writer lock protects each output run. Completed physical results
survive an interruption between scoring and logical publication. A process killed
before a result is durably published may need to recompute that unfinished call.

Each question reports winners under G-only, pure C and gold-aware ordering at
prefixes 8/16/32, their G/S/C and admissibility, strict G/C disagreements among
admissible pairs, quartet effects/reversals/productivity, fallback counts, distinct
physical masks, and per-call timing. All-pair comparisons share observations and
must not be treated as independent samples or unique causal labels. The 17 paired
question-level outcomes are the unit of the initial comparison.

R/A require 1,088 logical observations and 945 distinct static masks. Adding
adaptive requires 544 logical observations; its first eight per question are
already in A. Thus all three arms require **1,632 logical masked observations**
and at most **1,353 unique acquisition masks** before incidental additional reuse.
Report actual physical calls and wall time alongside this logical budget.

Every new question checkpoint additionally runs three parity likelihood evaluations
(one full context, the initial A mask through the persistent and legacy paths) and
one full-context generation. Those are common setup checks, recorded separately
from the 32-per-arm acquisition budget. The standalone smoke repeats one masked
score twice. Resume may repeat setup checks when a new reader process is needed.
No setup timing or extra check may be hidden in a claimed acquisition saving.

After scoring, decoded answer review remains necessary: use the union of the
three final winners from each arm plus the original three predetermined controls
(shared slot 0, R slot 1, A slot 2), at most 12 unique masked generations per
question. The scorer does not execute that later decoded-review stage. The
experiment cannot claim corrected answer accuracy from likelihood wins alone.

## Admission and current limits

Use the authentic execution override Qwen/Qwen2.5-VL-7B-Instruct revision
cc594898137f460bfe9f0759e9844b3ce807cfb5, Transformers 4.49.0, bfloat16, SDPA,
slow processor, greedy max_new_tokens=128 and repetition_penalty=1.05. The old
top-level Qwen2-VL fields in the historical config are not this reader.

Before acquisition, validate exact prompt and prefill hashes, four image grids,
10,032 original visual positions, sorted retained original IDs, B_input physical
trace, all-keep accepted/fixed-self likelihoods (max absolute error <=1e-4), a
masked persistent-versus-legacy parity check, and exact original generated token
IDs plus terminal EOS. Failure stops the run rather than silently widening a
tolerance. A full scoring command requires a matching successful one-question
smoke receipt and the same code, package and runtime identity.

Local tests verify controller behavior, geometry, journal recovery and mocked
adapter contracts. They do not prove real Qwen numerical parity or H200 memory
availability. No live H200 environment survey, installation, file transfer, model
load or GPU run was performed in steps 1–3. The saved environment inventory is
historical. Use the read-only doctor and current CoRAL survey before executing
step 4; the exact pinned stack is described in the handoff.
