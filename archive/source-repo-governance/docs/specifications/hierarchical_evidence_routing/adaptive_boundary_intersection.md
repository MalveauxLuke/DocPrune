# Adaptive Boundary Intersection

## Authoritative inference order

This procedure applies every time the system makes an indiscriminate geometric
split. The hierarchy stops after the quadrant level. Traversal may be
breadth-first by scoring the active candidates at one level before expanding
them, although correctness does not depend on choosing breadth-first instead of
depth-first.

For one split:

- First create `raw_A` and `raw_B` while preserving every complete routing
  atom. Any atom crossing the cut belongs to both raw children.
- If `raw_A intersection raw_B` is nonempty, use `A = raw_A` and `B = raw_B`
  without additional extension.
- If the raw children are disjoint, `A` is the first child extended by one
  adjacent DeepSeek routing atom from the other side of the cut, and `B` is
  the second child extended in the opposite direction.
- `C = A intersection B` is their shared boundary intersection.

The scoring order is fixed:

1. Create and score the two routing candidates `A` and `B` first.
2. If exactly one scores `yes`, descend through that candidate normally. Do not
   create or score `C`.
3. If both score `no`, stop those branches and retain or fall back to their
   parent.
4. Only if both `A` and `B` score `yes`, create and score `C`.
5. If `C` scores `no`, stop the `C` branch; `A` and `B` remain the previously
   positive branches.
6. If `C` scores `yes`, retain `C` as a positive branch and then create and
   score the exclusive candidates `A - C` and `B - C`.
7. Descend through every candidate that scores `yes`. Stop every candidate that
   scores `no`.
8. Repeat this procedure at the next indiscriminate split until the quadrant
   level is reached.

## Edge case: self-reproducing intersections

A brochure-like page may contain two large images that both span the height of
the page. A horizontal split can then produce:

```text
A = {Image 1, Image 2}
B = {Image 1, Image 2}
C = A intersection B = {Image 1, Image 2}
```

Scoring and recursively splitting `C` with the same horizontal operation would
recreate the same candidate indefinitely. A four-quadrant layout can produce a
similar failure whenever naturally crossing atoms or conditional one-unit
extension prevent a split from reducing candidate membership.

The resolution is mandatory:

1. Fingerprint candidates by `page_id + sorted(member_unit_ids)`.
2. Deduplicate candidates before scoring. If `A` and `B` are identical, score
   them once and do not create an identical `C`.
3. Require strict progress: every descent must reduce the member-unit set,
   advance to the next geometric level or axis, or transition to typed atomic
   candidate generation.
4. Never run an intersection through the same split-and-extension operation
   that created it.
5. Advance a half-level intersection to the vertical quadrant split.
6. Advance a quadrant-level intersection to fine-grained image, figure, table,
   and text candidates.
7. If a next-axis split still produces identical candidates, stop geometric
   splitting and use the underlying DeepSeek units as candidates.
8. Never score empty `A - C` or `B - C` candidates.
9. Cache scores by candidate fingerprint and maintain a visited-fingerprint
   set.
10. Retain the page-to-halves-to-quadrants depth limit and a per-query call
    budget as safety bounds.

For two side-by-side images spanning the horizontal midpoint, the horizontal
split is marked uninformative and the router advances vertically:

```text
left  = {Image 1}
right = {Image 2}
```

If the vertical split cannot separate them either, the router stops geometric
descent and exposes `Image 1` and `Image 2` as typed atomic candidates. The call
budget is a final guard; deduplication and strict progress prevent the loop by
construction.

## Decision history: unconditional boundary extension

The original procedure extended both children by one adjacent routing atom
after every split. It did so even when whole-atom preservation had already
placed a crossing atom in both raw children.

That unconditional rule is preserved in the original proposal below as a
paper trail and remains a useful ablation. It is no longer the default. The
authoritative procedure now suppresses additional extension whenever the raw
children already overlap and applies the one-atom repair only when they are
disjoint.

## Original idea

The following proposal is preserved as originally written so that later design
work does not silently change its intended inference behavior.

> but now weve increased out number of reranker calls by every time we do a split. what about this. we add some heuristics. we split a page top and bottom. imaginge
>
> IMAGE
>
> \_\_\_\_\_\_\_
>
> Text A
>
> now we have two candidates top half and bottom half. we can add text A to top half if an image or figure or even anything is near the edge. every time we extend the cut by one segment.  so top half gets image and text A. bottom half gets text A and image. assume there is more info on the page than what we just dispalyed. now both of them are candidates. if Top half scores 1 but bottom half doesnt we proceed as normal and vice versa. we know the info is not in image and text A because bottom scored and top didint. but if both scored well then we know something is up. now instead of proceeding as normal we add a third candidate which is the union of top and bottom half. we can even subtract this from top and bottom if we want. that might be better. now we score top half then bottom half then third candidate. we can score third candidate first, if it scores high we know the segment is in the intersection and we can proceed as normal. then we score top and bottom if yes for both we can contineu or just retunr like thw hole page if we deem necessary. if both top and bottom score low we know where the true segment is.
