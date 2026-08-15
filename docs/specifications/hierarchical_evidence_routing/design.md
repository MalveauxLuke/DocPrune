# Hierarchical Evidence Routing

## Status

This document defines the proposed hierarchical evidence-routing method.

It specifies the page-to-half-to-quadrant traversal and the adaptive
boundary-intersection procedure. Fine-grained candidate generation after the
quadrant level remains an open design question.

## Purpose

The system should reduce irrelevant page content before answer generation
without assuming that semantic segmentation can be solved perfectly in advance.

This is not permanent page segmentation. It is query-conditioned evidence
routing.

The system:

- begins with a retrieved page;
- divides it into progressively smaller spatial candidates;
- uses a binary reranker to decide which branches remain relevant;
- preserves larger parent candidates as fallbacks;
- permits several branches to remain active for multi-evidence queries;
- stops indiscriminate geometric splitting at the quadrant level.

## Core principle

A geometric cut must never crop or destroy a routing atom.

For text, the preferred routing atom is a complete semantic segment built from
one or more DeepSeek-OCR-2 units. For visuals, the routing atom is a complete
image, figure, or table. A raw DeepSeek paragraph or conservative paragraph/list
group is used only when no reliable semantic text segment exists.

The cut only determines which complete routing atoms are assigned to each
candidate. A routing atom that intersects a boundary remains whole.

If one or more routing atoms cross a boundary, those complete atoms are already
shared by both children and form the natural boundary overlap. No additional
one-atom extension is applied. Only when the raw children are disjoint does the
system protect potentially related units by extending each child by one routing
atom across the cut.

## Inputs

Each page provides:

- the original page image;
- DeepSeek-OCR-2 atomic units;
- deterministic semantic text segments and their member-unit IDs;
- the complete bounding box for each unit;
- unit type;
- OCR text, when available;
- DeepSeek reading order;
- the user query.

The reranker evaluates a query-candidate pair and returns a binary decision:

- `yes`: the candidate contains information relevant to answering the query;
- `no`: the candidate does not contain relevant information.

A `yes` decision means relevance. It does not necessarily prove that the
candidate contains all evidence required to answer a multi-evidence query.

## Terminology

### Parent

The candidate being divided.

### Child

One of the two candidates created by an indiscriminate geometric split.

### Active branch

A candidate that scored `yes` and remains eligible for further traversal.

### Terminated branch

A candidate that scored `no`. The system does not descend through it.

### DeepSeek unit

A complete atomic unit returned by DeepSeek-OCR-2, including its original
bounding box, type, content, and reading order.

DeepSeek units remain the immutable source representation and provenance layer.
They are not necessarily the units used by deterministic geometric routing.

### Semantic text segment

A deterministic logical group of one or more DeepSeek text units. Its geometry
is the union of its actual member boxes, not a rectangular envelope spanning
whitespace or unrelated content.

### Routing atom

An indivisible unit used by deterministic page-to-half and half-to-quadrant
routing:

- a complete semantic text segment when one is reliable;
- a complete image, figure, or table;
- a raw DeepSeek paragraph or conservative paragraph/list group only when no
  reliable semantic text segment exists.

Every routing atom retains the IDs of its underlying DeepSeek units.

### Indiscriminate split

A geometric split that does not claim to follow semantic boundaries.

The system uses two levels of indiscriminate splitting:

1. Page into top and bottom halves.
2. Active half-scale candidates into left and right quadrant-scale candidates.

### Extended child

A raw child that originally shared no crossing routing atom with its sibling
and therefore also contains the single selected routing atom from the opposite
side of the cut. When crossing atoms already exist, the raw children are used
without extension.

### Boundary intersection

The routing atoms shared by the two routing children, whether that overlap is
created naturally by a crossing atom or conditionally by one-atom extension.

For routing children `A` and `B`:

```text
C = A ∩ B
```

## Semantic preprocessing and routing policy

Semantic text segments are constructed once during preprocessing and cached.
This construction requires no query-conditioned reranker calls. Both the raw
DeepSeek units and their semantic-group membership are retained.

Deterministic geometric routing operates on complete semantic segments rather
than splitting them into paragraphs. This is required because evidence may be
compositional: a heading can supply the year, one paragraph can define an
entity, and another can provide the requested result. The complete section may
score `yes` even when every isolated paragraph scores `no`.

The reverse reconstruction is also not identifiable with the current binary
objective. Once a paragraph scores `yes`, every larger candidate containing it
may also score `yes`, even when the added paragraphs contribute nothing. The
reranker therefore cannot reliably reconstruct a useful semantic section by
scoring supersets of a known-positive paragraph.

For these reasons:

1. Reliable semantic segments are formed before geometric routing.
2. A semantic segment remains indivisible through halves and quadrants.
3. A segment whose member-box union crosses a cut is available in both raw
   children and therefore appears in their intersection. Its presence suppresses
   additional one-atom extension for that split.
4. A positive semantic segment is a terminal text candidate. The normal path
   does not descend into its raw paragraphs.
5. Raw paragraphs remain available for provenance, diagnostics, ablations, and
   fallback pages where no reliable semantic segment exists.

## Hierarchy

The basic spatial hierarchy is:

```text
page
├── top half
│   ├── top-left quadrant
│   └── top-right quadrant
└── bottom half
    ├── bottom-left quadrant
    └── bottom-right quadrant
```

Only branches that score `yes` are descended.

More than one branch may remain active.

The system stops indiscriminate geometric splitting after reaching
quadrant-scale candidates.

## Level 1: Page to top and bottom halves

The page is divided by a horizontal midpoint:

```text
              PAGE
             /    \
       TOP HALF   BOTTOM HALF
```

The split is not required to pass through whitespace. It may pass through or
between underlying DeepSeek boxes because it is only being used to assign
complete routing atoms to spatial candidates.

### Whole-atom membership

No routing atom is cropped.

If a routing atom crosses the horizontal boundary:

- retain the complete routing atom;
- make it available to both raw children;
- do not apply an additional one-atom extension at that split.

For a semantic segment, crossing and overlap are computed from the union of its
actual member boxes. A large rectangular envelope must not be used.

For a routing atom that does not cross the boundary, its initial side is
determined from its geometry relative to the two halves. The exact membership
calculation remains an implementation decision.

Possible methods include:

- maximum intersection-over-union;
- maximum intersection area;
- intersection-over-unit coverage;
- bounding-box center.

### Conditional one-atom boundary extension

After initial assignment, compute the naturally shared crossing set:

```text
shared_crossing = raw_top ∩ raw_bottom
```

If `shared_crossing` is nonempty:

- use `A = raw_top` and `B = raw_bottom`;
- do not copy any additional adjacent atom across the cut;
- allow the naturally crossing atoms to become `C` only if both `A` and `B`
  later score `yes`.

If `shared_crossing` is empty, apply one-atom extension:

- the top child receives the first adjacent routing atom from the bottom;
- the bottom child receives the last adjacent routing atom from the top.

This creates two extended candidates:

```text
A = extended top half
B = extended bottom half
```

Example:

```text
Additional content in top half

IMAGE
---------------- horizontal cut
Text A

Additional content in bottom half
```

Because the raw children in this example are disjoint, the extended candidates
become:

```text
A = top content + IMAGE + Text A
B = IMAGE + Text A + bottom content
```

The shared boundary intersection is:

```text
C = IMAGE + Text A
```

However, `C` is not created or scored immediately.

## Authoritative scoring order

The scoring order is fixed.

### Step 1: Score both routing children

Score:

```text
score(A)
score(B)
```

The intersection candidate does not yet require a reranker call.

### Step 2: Exactly one child scores yes

If:

```text
A = yes
B = no
```

descend through `A` and terminate `B`.

If:

```text
A = no
B = yes
```

descend through `B` and terminate `A`.

Do not create or score `C`.

The different decisions indicate that the shared boundary content was not
sufficient to make both routing children positive.

### Step 3: Both children score no

If:

```text
A = no
B = no
```

terminate both child branches.

Retain or return the parent candidate as a fallback if the parent had
previously scored `yes`.

This outcome may indicate:

- a reranker error;
- a thresholding error;
- distributed evidence that the children do not preserve adequately;
- a false-positive parent;
- evidence that cannot be localized through this split.

### Step 4: Both children score yes

If:

```text
A = yes
B = yes
```

the result is ambiguous.

Possible explanations include:

- the relevant evidence lies in the shared boundary region;
- independent relevant evidence exists on both sides;
- the query requires evidence from both sides;
- the reranker produced two broad positive decisions.

Only in this case create:

```text
C = A ∩ B
```

Then score `C`.

### Step 5: Intersection scores no

If:

```text
C = no
```

terminate the `C` branch.

The previously positive `A` and `B` branches remain active. The evidence is
likely outside the shared intersection, distributed across the two sides, or
affected by reranker noise.

Continue routing the positive branches.

### Step 6: Intersection scores yes

If:

```text
C = yes
```

retain `C` as an active candidate.

Then construct and score the exclusive portions:

```text
A_exclusive = A − C
B_exclusive = B − C
```

Score:

```text
score(A_exclusive)
score(B_exclusive)
```

For each resulting candidate:

- `yes`: retain and descend;
- `no`: terminate.

This determines whether relevant evidence exists only in the intersection or
also outside it.

The system may therefore retain:

- only `C`;
- `C` and `A_exclusive`;
- `C` and `B_exclusive`;
- all three candidates.

Multiple retained candidates are permitted because real queries may require
evidence from several locations.

## Level 2: Active regions to quadrants

Every active half-scale branch is divided vertically into left and right
candidates.

For the ordinary top and bottom branches, this produces:

```text
                    PAGE
             ┌────────┴────────┐
          TOP HALF         BOTTOM HALF
          /      \          /        \
     TOP-LEFT TOP-RIGHT BOTTOM-LEFT BOTTOM-RIGHT
```

For every active candidate:

1. Split it vertically.
2. Preserve complete routing atoms, including every atom crossing the cut in
   both raw children.
3. Compute the naturally shared crossing set.
4. If that set is nonempty, use the raw children without additional extension.
5. If the raw children are disjoint, extend the left child by one adjacent
   routing atom from the right and the right child by one adjacent routing atom
   from the left.
6. Score both routing children.
7. Apply the same intersection procedure if both score `yes`.
8. Descend through `yes` candidates.
9. Terminate `no` candidates.

Positive auxiliary candidates created by the intersection procedure may also
continue to quadrant-scale localization.

## Progress, deduplication, and loop prevention

Naturally crossing atoms or conditional one-atom extension can make two
children contain the same underlying DeepSeek member units as each other or as
their parent. Without an explicit progress rule, an intersection can reproduce
itself indefinitely.

Every candidate therefore has a canonical fingerprint:

```text
fingerprint(candidate) = page_id + sorted(member_unit_ids)
```

Because indiscriminate cuts never crop DeepSeek units, two candidates on the
same page with the same member-unit set contain the same evidence and share one
cached reranker score.

The following rules are mandatory:

1. Canonicalize and deduplicate candidates before making reranker calls.
2. If routing children `A` and `B` have the same fingerprint, score that
   candidate at most once. Do not create the identical intersection `C`.
3. Declare a split uninformative when it produces no distinct proper
   refinement of its parent.
4. Every descent must either reduce the member-unit set, advance to the next
   geometric level or axis, or transition to typed routing-candidate
   generation.
5. An intersection may not repeat the same split-and-extension operation that
   created it. A half-level intersection advances to the vertical
   quadrant-level split. A quadrant-level intersection advances to
   fine-grained candidate generation.
6. If the next-axis split is also uninformative, stop geometric routing and
   expose the intersection's complete images, figures, tables, reliable
   semantic text segments, and paragraph fallbacks where required.
7. Do not create or score empty `A − C` or `B − C` candidates.
8. Maintain a visited-fingerprint set and reuse every cached score.
9. Enforce the page-to-halves-to-quadrants depth limit and a configured
   per-query call budget as final safety bounds. The budget does not replace
   the strict-progress rules.

For example, consider two side-by-side images that both span the page's
horizontal midpoint:

```text
A = {Image 1, Image 2}
B = {Image 1, Image 2}
```

Here `A`, `B`, and their would-be intersection are identical. The router scores
the shared candidate once, marks the horizontal split uninformative, and
advances to the vertical split. If that split separates the images, routing
continues with `{Image 1}` and `{Image 2}`. If it does not, the router moves
directly to typed atomic candidate generation.

## Geometric stopping condition

Indiscriminate geometric traversal stops after quadrant-scale localization:

```text
page → halves → quadrants → stop
```

The system does not automatically continue to:

- eighths;
- sixteenths;
- progressively smaller fixed rectangles;
- pixel-level regions.

Further localization requires a separate fine-grained candidate-generation
method.

## Breadth-first and depth-first scheduling

The routing rules do not require one traversal schedule.

### Breadth-first option

Score every active candidate at the current hierarchy level before splitting
any candidate at the next level.

Example:

1. Score top and bottom.
2. Resolve any half-level intersection.
3. Determine all active half-level candidates.
4. Split and score their quadrant-level children.

This may make page-level routing behavior easier to inspect and batch.

### Depth-first option

Immediately descend through a positive candidate before processing its sibling.

This may reduce latency when only one branch remains positive.

The choice affects scheduling and batching but does not change the meaning of
`yes`, `no`, intersection, or subtraction.

Breadth-first traversal is currently the clearer default, but this remains an
implementation choice.

## Why extension occurs only for disjoint raw children

The primary boundary-specific failure occurs when potentially related routing
atoms are immediately adjacent across the cut.

Example:

```text
IMAGE
---------------- cut
Text A
```

When the raw children are disjoint, the image and `Text A` appear in separate
children.

With one-atom extension, both routing candidates contain the possible pair.

When an atom physically crosses the cut, however, whole-atom preservation
already places it in both raw children. That shared atom is a natural boundary
intersection. Extending again would add unrelated neighboring content, create
more identical or near-identical candidates, and spend calls without repairing
an additional boundary loss.

If another routing atom is two positions below the boundary, it remains safely
preserved by the original bottom child. It has not been removed from the
hierarchy.

Therefore:

- naturally crossing atoms are shared without additional extension;
- only when the raw children are disjoint is the immediately adjacent routing
  atom copied across the boundary;
- routing atoms farther away remain preserved by their original child;
- a routing atom physically crossing the cut remains whole and is available to
  both;
- the parent remains available for distributed evidence.

Extending a naturally overlapping split, or extending a disjoint split by more
than one routing atom, would increase candidate noise and reranker cost without
addressing an additional immediate boundary-loss problem.

## Multi-evidence behavior

The system does not assume that exactly one candidate is correct.

If several branches score `yes`, the controller may retain several branches.

Examples include:

- a query requiring information from two quadrants;
- an image in one region and explanatory text in another;
- several visual panels contributing to one answer;
- evidence distributed across the page.

The system must not arbitrarily discard one positive branch solely because
another branch also scored positively.

The original parent remains available as a fallback when:

- all children score `no`;
- evidence appears distributed;
- finer localization becomes unreliable;
- the answerer requires wider page context.

The final policy for choosing between multiple children, their union, and the
parent remains an open decision.

## Reranker-call efficiency

An ordinary split requires two reranker calls:

```text
score(A)
score(B)
```

The intersection adds a third call only when both children score `yes`:

```text
score(C)
```

The exclusive candidates add two more calls only when `C` also scores `yes`:

```text
score(A − C)
score(B − C)
```

Therefore, the additional work is conditional:

- ordinary unambiguous split: two calls;
- double-positive split: three calls;
- positive intersection requiring exclusivity testing: five calls.

The method spends additional computation only when the initial routing result
is ambiguous.

## Fine-grained candidate generation

Fine-grained candidate generation begins only after the system reaches a
relevant quadrant-scale candidate.

Text and visual routing diverge at this point. A reliable semantic text segment
is already a terminal text candidate: if it scores `yes`, return the complete
segment rather than descending into its paragraphs. Fine-grained visual
candidate generation remains open.

Possible candidates include:

- complete semantic text segments intersecting the quadrant;
- paragraph/list fallbacks only when no reliable semantic segment exists;
- individual complete visual atoms;
- an image or figure alone;
- an image plus the preceding text unit;
- an image plus the following text unit;
- an image plus both neighboring text units;
- adjacent text-unit groups;
- geometry-based visual-text combinations;
- reading-order-based combinations;
- candidates constrained by detected frames or separators.

These candidates are hypotheses. Candidate generation does not permanently
declare that an image and a text unit form a true caption pair.

The reranker evaluates each query-candidate relationship.

## Possible role of Hough transforms

Hough-derived lines may eventually contribute optional geometric evidence.

Possible uses include:

- detecting strong separators that reduce the plausibility of cross-line
  associations;
- detecting shared rectangular frames around images and text;
- identifying panels, cards, or form regions.

Hough transforms should not be treated as semantic image-caption linkers.

The decision to include Hough-derived evidence remains open and should be
evaluated against simpler geometry and reading-order baselines.

## Non-goals

This design does not:

- require perfect semantic segmentation before retrieval;
- require known image-caption relationships at inference time;
- force every image to have one caption;
- force one-to-one image-text matching;
- crop routing atoms at geometric boundaries;
- split a reliable semantic segment into paragraphs during normal routing;
- permanently discard the page after descending;
- assume that one query has only one evidence location;
- split geometrically below the quadrant level;
- define the final fine-grained candidate inventory.

## Open implementation decisions

The following decisions remain unresolved:

1. Exact binary reranker threshold.
2. Whether routing uses hard labels alone or retains yes/no logits.
3. Exact bounding-box membership calculation.
4. Tie-breaking when a routing atom overlaps both children equally.
5. When conditional extension is required, how the adjacent routing atom
   across a cut is selected when reading order and geometry disagree.
6. Breadth-first versus depth-first execution.
7. Batching strategy for distinct uncached candidates.
8. Exact per-query reranker-call budget.
9. Final policy for returning multiple branches versus their parent.
10. Fine-grained candidate generation after quadrants.
11. Whether Hough separators or frames provide useful additional evidence.
12. How to detect and handle reranker contradictions.
13. How to determine whether a positive candidate contains complete or only
    partial evidence.

## Required invariants

Any implementation must preserve these invariants:

1. Routing atoms are never cropped by indiscriminate splits.
2. A branch scoring `no` is not descended.
3. A branch scoring `yes` remains eligible for descent.
4. `C` is created and scored only when both routing children score `yes`.
5. `A − C` and `B − C` are scored only when `C` scores `yes`.
6. More than one positive branch may remain active.
7. Parent and page candidates remain available as fallbacks.
8. Indiscriminate geometric splitting stops after quadrants.
9. Fine-grained image-text relationships are hypotheses, not assumed ground
   truth.
10. The original DeepSeek units and their provenance remain recoverable.
11. Identical candidate fingerprints share one cached reranker score.
12. Every recursive descent makes strict progress by reducing membership,
    advancing the geometric level or axis, or entering typed routing-candidate
    generation.
13. An intersection never repeats the split-and-extension operation that
    created it.
14. Empty exclusive candidates are never scored.
15. Additional one-atom extension occurs only when the two raw children have
    no naturally shared crossing routing atom.

## Conceptual pseudocode

```text
route(parent, level, visited):

    if level is deeper than quadrants:
        return parent as an active fine-grained input

    if fingerprint(parent) is in visited:
        return typed_routing_candidates(parent)

    visited.add(fingerprint(parent))

    raw_A, raw_B = geometric_split(parent)

    shared_crossing = intersection(raw_A, raw_B)

    if shared_crossing is not empty:
        A = raw_A
        B = raw_B
    else:
        A = extend_by_one_routing_atom(raw_A, across_boundary_from=raw_B)
        B = extend_by_one_routing_atom(raw_B, across_boundary_from=raw_A)

    if fingerprint(A) == fingerprint(B):
        shared_score = rerank_with_cache(query, A)

        if shared_score == no:
            return fallback(parent)

        return advance_axis_or_atomize(A, next_level, visited)

    score_A = rerank_with_cache(query, A)
    score_B = rerank_with_cache(query, B)

    if score_A == no and score_B == no:
        return fallback(parent)

    if score_A == yes and score_B == no:
        return route(A, next_level, visited)

    if score_A == no and score_B == yes:
        return route(B, next_level, visited)

    C = intersection(A, B)
    score_C = rerank_with_cache(query, C)

    if score_C == no:
        return {
            route(A, next_level, visited),
            route(B, next_level, visited)
        }

    A_exclusive = subtract(A, C)
    B_exclusive = subtract(B, C)

    active = {C}

    if A_exclusive is not empty:
        score_A_exclusive = rerank_with_cache(query, A_exclusive)

        if score_A_exclusive == yes:
            active.add(A_exclusive)

    if B_exclusive is not empty:
        score_B_exclusive = rerank_with_cache(query, B_exclusive)

        if score_B_exclusive == yes:
            active.add(B_exclusive)

    return route_all_at_next_level_or_atomize(active, next_level, visited)
```

## Evaluation requirements

The method should be compared with:

- full-page answer generation;
- non-overlapping halves and quadrants;
- the legacy unconditional one-atom-extension policy;
- conditional one-atom extension only when raw children are disjoint;
- overlapping halves and quadrants without intersection testing;
- the complete adaptive intersection procedure.

Measurements should include:

- answer accuracy;
- evidence recall;
- reranker calls per query;
- average number of active branches;
- page-fallback frequency;
- boundary-related evidence losses;
- frequency with which natural crossing overlap suppresses extension;
- multi-evidence preservation;
- smallest successful candidate;
- cases rescued or harmed by the intersection procedure.

The primary question is whether hierarchical routing improves answer generation
relative to presenting the complete page while preserving evidence that lies
across geometric boundaries. The extension ablation also tests whether the
conditional policy preserves those benefits with less noise and fewer redundant
candidates than the legacy unconditional policy.

## Decision history: unconditional one-atom extension

The original boundary rule applied one-atom extension after every split:

- the first child received one adjacent routing atom from the second;
- the second child received one adjacent routing atom from the first.

That rule applied even when whole-atom preservation had already placed one or
more boundary-crossing atoms in both raw children. For example:

```text
raw_A = {Top content, Crossing segment}
raw_B = {Crossing segment, Bottom content}

legacy A = {Top content, Crossing segment, Bottom content}
legacy B = {Top content, Crossing segment, Bottom content}
```

The unconditional rule was superseded because the crossing segment already
provides the intended overlap. Copying another atom in that case adds noise,
can make the two candidates identical to each other or their parent, and may
create unnecessary reranker calls.

The adopted rule preserves the same one-atom repair when the raw children are
disjoint, but suppresses additional extension when natural crossing overlap is
already present. The unconditional policy remains documented here and should
remain available as an evaluation ablation; it is no longer the authoritative
default.

## Decision history: original paragraph-first plan

The original design used raw DeepSeek paragraphs as the indivisible units for
page-to-half and half-to-quadrant routing. Semantic sections were retained only
as a later candidate view. After a positive paragraph was found, the router
could attempt to expand back to its complete semantic section.

That plan was replaced for two related reasons:

1. **Compositional evidence can disappear after paragraph splitting.** A
   heading and several paragraphs may jointly answer a query while no isolated
   paragraph contains enough context to score `yes`.
2. **Superset scoring cannot validate reconstruction.** Once a paragraph scores
   `yes`, the complete section containing it is also likely to score `yes`
   whether or not the surrounding paragraphs are relevant. The current binary
   reranker cannot determine their marginal value by rescoring that superset.

The adopted design therefore preprocesses reliable semantic sections and uses
them as indivisible text routing atoms. Raw paragraphs remain the immutable
source representation and serve as fallbacks only when reliable semantic
grouping is unavailable. The geometric tree, adaptive `A`/`B`/`C` intersection
order, score caching, strict-progress rules, quadrant depth limit, visual atoms,
and parent fallbacks are unchanged from the original plan.
