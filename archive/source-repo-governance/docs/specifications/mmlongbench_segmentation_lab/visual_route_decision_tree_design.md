# Visual Route Decision Tree

## Status and authority

This document specifies an additive visualization for the local MMLongBench
segmentation lab. It does not change segmentation, routing candidates, human
decisions, deduplication, fallback behavior, or the audit schema.

The routing behavior in
[`../hierarchical_evidence_routing/design.md`](../hierarchical_evidence_routing/design.md)
remains authoritative. The lab contract in [`design.md`](design.md) remains
authoritative for the page viewer and manual-routing state machine. If this
visualization specification conflicts with either document, those documents
win.

Status: implemented and verified locally on 2026-07-23.

## Goal

Make the complete routing history visually understandable without requiring a
user to reconstruct branches from the raw audit trail.

The visualization must show every action recorded in the audit trail and the
candidate relationships that produced those actions. It is a read-only view
of existing routing state, except that selecting a tree node changes which
candidate is highlighted on the reference page.

## Chosen interaction

The existing candidate-routing panel has two views:

- **Live route**: the current decision-making candidate cards.
- **Visual route**: the complete visual decision tree.

The two views occupy the same panel and use the same available width and
height. Switching views must not:

- resize, rescale, cover, or close the reference page;
- resize or cover the audit drawer;
- change any routing decision or candidate;
- change the page/routing workspace divider; or
- discard the live-route scroll position.

The first-ever default is **Live route**. After the user switches views, that
choice persists across pages, documents, and reloads as a workspace preference.
It is not part of routing-session identity and must not appear in exported
review state.

The visual tree stays synchronized after every decision and automatic routing
transition, even while Live route is visible. The application never switches
views automatically.

## Visual-tree contents

### Candidate nodes

Display one node for every unique candidate fingerprint known to the active
session, including:

- the permanent page fallback;
- pending and positive active candidates;
- candidates removed after a `no` decision;
- intersections and nonempty exclusive candidates;
- typed terminal candidates; and
- candidates reused through fingerprint deduplication.

Each candidate node contains:

- the existing miniature page-position preview used by live candidate cards;
- candidate kind;
- current status;
- member-unit count; and
- the audit actions associated with that candidate.

The initial implementation does not require newly authored explanatory
captions. Existing audit event text is sufficient.

### Audit actions

Every audit event must appear exactly once in the Visual route view. Preserve
its global chronological position with a visible sequence number.

Candidate-associated events appear within or directly beside the corresponding
candidate node. Supported text includes the existing event types:

- `candidate revealed`;
- `human decision`;
- `candidate removed`;
- `cached decision reused`;
- `axis advanced`;
- `split declared uninformative`;
- `intersection revealed`;
- `exclusive candidate revealed`; and
- `terminal candidates exposed`.

If a future audit event has no candidate ID, display it in a session-event lane
rather than dropping it.

### Branches and reuse

Parent-child candidate relationships form the primary tree edges. Branches
must preserve multiple simultaneous positive paths.

Fingerprint reuse does not create a duplicate candidate node. Instead, show a
reuse reference pointing to the existing node and retain the corresponding
`cached decision reused` event with its sequence number.

Preserve the attempted split structure around that reference. A deduplicated
child must still appear under its recorded horizontal or vertical split with
its attempted kind, such as `half bottom`, and indicate that it reused the
canonical page candidate. This reference is not a second candidate and is not
scored again.

The visualization must distinguish:

- ordinary parent-child revelation;
- a candidate removed by `no`;
- a candidate retained by `yes`;
- a pending candidate;
- terminal evidence;
- an uninformative split; and
- an axis advancement or cached reuse.

Exact colors and connector ornamentation are implementation details. Clarity
of the branch structure and inclusion of every event take priority over
decorative captions.

## Candidate selection and page highlighting

Every candidate node is selectable.

Selecting a tree node must use the same selected-candidate state as clicking a
Live route candidate card. It must immediately:

1. make that candidate the selected candidate;
2. render its complete member units through the existing Candidate membership
   overlay on the reference page;
3. update the reference details text; and
4. visually mark the selected tree node.

This selection must work for removed candidates and reused candidates, even
when they no longer appear as active Live route cards. Selecting evidence for
inspection must never restore a removed candidate or change its decision.

When a reuse reference is selected, highlight the canonical candidate node and
its evidence membership. The reuse event remains separately visible in its
original chronological position.

Switching back to Live route preserves the selected candidate whenever that
candidate has a live card. If the selected candidate is removed and therefore
has no live card, Live route keeps the evidence selection but does not invent a
replacement card.

## Layout and scrolling

Visual route renders inside the existing candidate-routing panel. The panel
retains its current dimensions and independent overflow behavior.

The tree may scroll horizontally and vertically inside that panel. Large trees
must not increase the desktop workspace height or force the reference page to
rescale. The implementation may arrange nodes by routing depth and reveal
order, but it must not imply a false parent-child relationship merely to obtain
a compact layout.

The chronological audit drawer remains visible. Its presentation uses event
names, human-readable candidate types, and timestamps; internal IDs remain in
session state and JSON export rather than occupying review space. Visual route
supplements this reproducibility trail.

## Data flow

Visual route derives its model only from existing browser-session state:

```text
session.candidates
session.expansions
session.audit
session.graph
        |
        v
visual-route projection
        |
        v
candidate nodes + edges + ordered audit actions
```

The projection is disposable view state. It must not mutate or reinterpret the
routing controller.

Candidate identity comes from the existing canonical fingerprint. Parent-child
edges come from candidate parent IDs and recorded expansion references. Reuse
references come from the same expansion/fingerprint information that currently
produces `cached decision reused`.

## Persistence

Persist only the selected Live route/Visual route view under a new,
schema-versioned workspace key:

```text
mmlongbench-segmentation-lab:candidate-view:v1
```

Do not add the view preference, tree layout, scroll positions, or selected
visual node to routing-session export/import.

## Accessibility

- Implement Live route and Visual route as keyboard-operable tabs with an
  explicit selected state.
- Candidate tree nodes must be keyboard selectable.
- Status and audit actions must remain understandable without color.
- The miniature preview is supplementary; candidate kind, decision status,
  unit count, and audit events remain textual.

## Failure behavior

- An audit event with an unknown candidate ID remains visible in the
  session-event lane.
- A candidate with missing preview geometry still renders its textual node.
- A reuse reference that cannot resolve its canonical candidate remains visible
  as an unresolved reuse event rather than being silently removed.
- Visual-route rendering errors must not prevent Live route decisions or audit
  export.

## Implementation surface

Expected modifications:

- `viewer/mmlongbench_segmentation_lab/index.html`
- `viewer/mmlongbench_segmentation_lab/styles.css`
- `viewer/mmlongbench_segmentation_lab/app.js`
- `tests/test_mmlongbench_segmentation_lab_static.py`
- `tests/test_mmlongbench_segmentation_lab_integration.py`

No routing-graph or segmentation-strategy change is expected. No frontend
framework or production dependency is required.

## Acceptance criteria

1. Live route and Visual route occupy the same candidate-panel dimensions.
2. The first default is Live route and a manual view choice persists across
   pages, documents, and reloads.
3. Switching views never changes routing state, reference-page scale, workspace
   dimensions, or the audit drawer.
4. Visual route displays every audit event exactly once in chronological order.
5. Visual route displays every unique session candidate, including removed,
   terminal, and reused evidence.
6. Candidate branches reflect actual parent and expansion relationships.
7. Reused fingerprints reference one canonical candidate node instead of
   creating misleading duplicates.
8. Candidate nodes reuse the existing miniature candidate preview.
9. Clicking or keyboard-selecting any candidate node immediately highlights its
   complete evidence membership on the reference page.
10. Removed and reused candidates remain selectable for evidence inspection
    without changing their routing decisions.
11. The tree scrolls within the existing panel and never enlarges the desktop
    workspace or rescales the page.
12. Live route remains fully functional if Visual route cannot render.
