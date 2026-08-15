# Hierarchical Evidence Routing

Status: current research focus; design documented, unstructured OCR inputs
available locally, and production implementation not started.

Complete design:
[`docs/specifications/hierarchical_evidence_routing/design.md`](../../../docs/specifications/hierarchical_evidence_routing/design.md).

Boundary-intersection detail:
[`adaptive_boundary_intersection.md`](../../../docs/specifications/hierarchical_evidence_routing/adaptive_boundary_intersection.md).

## Core idea

Route query-conditioned evidence from page to halves to quadrants and then to
fine-grained OCR units. A parent remains available as fallback. Every complete
routing atom crossing a cut is included in both raw children, and this natural
overlap suppresses additional extension. Only disjoint raw children borrow one
adjacent atom per side, selected by the active geometry or reading-order mode.
When both routing children score yes, an adaptive boundary intersection
determines whether to continue locally or retain the parent/multiple branches.

Candidate fingerprints, cached scores, and a strict-progress invariant prevent
routing children or intersections from reproducing themselves. An
uninformative split advances to the next axis or directly to typed atomic
candidates; intersections never repeat the split that created them.

The earlier rule extended both children after every split. It is retained in
the canonical design's decision history and as an evaluation ablation, but the
conditional disjoint-only rule is authoritative.

No production router currently implements this design.

## Current experimental path

Use the 306 successful pages from the retained MMLongBench unstructured OCR run
to implement and visually test the segmentation and routing rules. Query-linked
gold-page navigation is a later extension after the rules stabilize. All
current work is local.
