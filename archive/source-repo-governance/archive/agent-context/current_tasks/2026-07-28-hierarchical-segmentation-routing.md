# Current Task State

## Goal

- Implement and use a modular local segmentation lab to test and refine the
  hierarchical evidence-routing rules on the retained unstructured
  MMLongBench DeepSeek-OCR-2 output.

## Scope

- Work locally on `main` on this computer.
- Use all 306 validated OCR pages from the 313-page unstructured MMLongBench
  run. The 7 audited OCR failures remain excluded.
- Let a human manually assign `yes` or `no` while the website executes the
  exact page-to-halves-to-quadrants routing state machine.
- Preserve every complete atom crossing a cut in both raw children and apply
  no additional extension at that split.
- When raw children are disjoint, compare geometry-based and
  reading-order-based selection for the one-atom-per-side boundary extension.
- Keep segmentation and terminal-candidate generation modular so later rule
  variants can be inspected without rebuilding the website architecture.
- Keep website implementation and segmentation-rule testing local. A separate,
  OCR-only SOL acquisition is now planned for the gold-page reranker baseline;
  SOL still does not perform segmentation, reranking, answering, or display.

## Canonical plans and inputs

- `docs/specifications/hierarchical_evidence_routing/design.md`
- `docs/specifications/hierarchical_evidence_routing/adaptive_boundary_intersection.md`
- `docs/specifications/mmlongbench_segmentation_lab/design.md`
- `docs/specifications/mmlongbench_segmentation_lab/implementation_plan.md`
- `docs/specifications/mmlongbench_segmentation_lab/hybrid_routing_resizable_workspace_plan.md`
- `docs/specifications/mmlongbench_segmentation_lab/visual_route_decision_tree_design.md`
- `docs/specifications/deepseek_semantic_segmentation.md`
- `docs/specifications/mmlongbench_gold_page_quadrant_reranker_experiment.md`
- `sol/task_spec/mmlongbench_gold_pages_deepseek_ocr2.md`
- `pilot_data/mmlongbench_ocr2_unstructured_313/`
- `sol_results/mmlongbench_ocr2_unstructured_313/20260717T182300Z/`
- `viewer/mmlongbench_ocr2/`

## Decisions made

- Treat segmentation as candidate generation rather than an irreversible
  page-wide partition.
- Never crop a DeepSeek unit during geometric splitting.
- Route page to halves to quadrants and stop indiscriminate geometry there.
- Follow the authoritative `A/B/C`, intersection, subtraction, deduplication,
  and strict-progress rules exactly.
- Preserve every positive branch and the page/parent fallback; only negative
  candidates leave the active board.
- Use `deepseek_semantic_v1` as the initial replaceable segmentation strategy.
- Preserve `deepseek_semantic_v1` as the comparison baseline. The additive
  `deepseek_semantic_hybrid_v1` strategy keeps uninterrupted semantic text
  sections whole, splits them into contiguous routing runs only when another
  DeepSeek unit interrupts their complete reading order, and creates no
  image-caption pairs.
- Treat natural crossing overlap as sufficient boundary protection. Invoke
  geometry or reading-order extension only for a disjoint split, as separate
  reproducible modes.
- Retain the former unconditional-extension policy in the canonical design's
  decision history and as a possible ablation, not as the default.
- Keep human target highlights visually useful but algorithmically separate
  from manual routing decisions.

## Potential improvements under consideration

- A full-page vertical rescue after two distinct horizontal children are both
  `no` is documented but not implemented. It would retain the page fallback,
  attempt exactly one orthogonal split with the existing atom/extension/
  deduplication rules, and stop if the vertical candidates are also negative
  or uninformative.
- The strongest proposed initial scope is known-positive or gold pages. Using
  the rescue for arbitrary retrieved pages could add unnecessary scores and
  false-positive exposure.
- This differs from existing axis advancement when a horizontal child reuses
  the page fingerprint; that strict-progress behavior is already implemented.

## Current state

- The routing and adaptive-boundary behavior are specified.
- The retained unstructured result contains 306 packaged OCR pages from 10
  documents and 7 explicitly excluded failures.
- The local segmentation-lab builder, static viewer, and integration coverage
  are implemented; the design and implementation plan remain authoritative.
- The additive hybrid strategy is implemented without changing the baseline.
  The desktop lab is viewport-bounded and has a persisted page/routing divider;
  resizing changes the visible page viewport rather than the rendered page
  dimensions.
- The toggleable Visual route decision tree is implemented in the existing
  Live route panel. It shows every audit action and candidate branch, persists
  the selected view across navigation and reloads, and lets every tree node,
  including removed candidates, highlight its membership on the reference
  page without changing routing state.
- The website remains local work. A separate SOL handoff is prepared, but not
  yet launched, to acquire DeepSeek-OCR-2 output for all valid MMLongBench gold
  pages required by the planned reranker baseline.

## Current blockers

- The segmentation-lab work is unblocked because its retained unstructured OCR
  inputs are already local. The separate gold-page reranker baseline cannot
  begin locally until its prepared SOL OCR acquisition is executed and the
  returned bundle is pulled back.

## Verified local smoke build (2026-07-21)

- `python -m pytest -q tests/test_mmlongbench_segmentation_lab_integration.py --tb=short`
- `python scripts/mmlongbench/build_segmentation_lab.py --pilot-root pilot_data/mmlongbench_ocr2_unstructured_313 --result-root sol_results/mmlongbench_ocr2_unstructured_313/20260717T182300Z --viewer-source viewer/mmlongbench_segmentation_lab --site-dir tmp/mmlongbench_segmentation_lab_task6_smoke --strategy deepseek_semantic_v1 --document 91521110100M_4K_UHD_Display_User_Manual_V1.1.pdf --page-id mmlongbench_page_94c4dea17c77f92613a4`
  produced `tmp/mmlongbench_segmentation_lab_task6_smoke/`.
- Final retained verification used
  `python scripts/mmlongbench/build_segmentation_lab.py --pilot-root pilot_data/mmlongbench_ocr2_unstructured_313 --result-root sol_results/mmlongbench_ocr2_unstructured_313/20260717T182300Z --viewer-source viewer/mmlongbench_segmentation_lab --site-dir tmp/mmlongbench_segmentation_lab_final_r2_20260721 --strategy deepseek_semantic_v1 --page-id mmlongbench_page_06715bbb9f895678bbe2 --page-id mmlongbench_page_04f2fe9d7c995d718a55 --page-id mmlongbench_page_289d0028a99c349a5c4d`
  and produced `tmp/mmlongbench_segmentation_lab_final_r2_20260721/`.

## Hybrid comparison smoke build (2026-07-23)

- `python scripts/mmlongbench/build_segmentation_lab.py --pilot-root pilot_data/mmlongbench_ocr2_unstructured_313 --result-root sol_results/mmlongbench_ocr2_unstructured_313/20260717T182300Z --viewer-source viewer/mmlongbench_segmentation_lab --site-dir tmp/mmlongbench_segmentation_lab_hybrid_r2_20260723 --strategy deepseek_semantic_v1 --strategy deepseek_semantic_hybrid_v1 --page-id mmlongbench_page_06715bbb9f895678bbe2`
  produced `tmp/mmlongbench_segmentation_lab_hybrid_r2_20260723/`.
- On `mmlongbench_page_06715bbb9f895678bbe2`, the hybrid produces disjoint raw
  halves. Geometry extension includes the upper image in the bottom candidate
  and the following text in the top candidate.

## Visual route full build (2026-07-23)

- `python scripts/mmlongbench/build_segmentation_lab.py --pilot-root pilot_data/mmlongbench_ocr2_unstructured_313 --result-root sol_results/mmlongbench_ocr2_unstructured_313/20260717T182300Z --viewer-source viewer/mmlongbench_segmentation_lab --site-dir tmp/mmlongbench_segmentation_lab_visual_route_full_r5_20260723 --strategy deepseek_semantic_v1 --strategy deepseek_semantic_hybrid_v1`
  produced a locally served comparison site with 10 documents, 306 pages, 7
  excluded OCR failures, and both segmentation strategies.
- The visual tree retains each attempted split by axis and kind even when a
  child fingerprint reuses an existing candidate. On
  `mmlongbench_page_e87ba29cc59d515b82e6`, it shows horizontal `half top` and
  `half bottom → reused page`, followed by vertical `half left → reused page`
  and `half right`.
- Live candidate cards, reference summaries, and audit entries now omit
  internal fingerprints and IDs. Exact identifiers remain in exported review
  JSON.

## Full hybrid review build (2026-07-23)

- `tmp/mmlongbench_segmentation_lab_hybrid_full_20260723/` contains all 306
  validated pages across 10 documents with both `deepseek_semantic_v1` and
  `deepseek_semantic_hybrid_v1`.
- The generated manifest, 306 page payloads, 306 page images, and both
  strategies on every page were verified.
- Focused integration/static verification passed: `44 passed`.
- The local review site is served at `http://127.0.0.1:8015/`.

## Research sequence

1. Implement the local page-centric segmentation lab over the retained
   unstructured OCR result.
2. Use it to inspect the semantic routing atoms, verify that natural crossing
   overlap suppresses extra extension, and compare geometry versus reading
   order for disjoint splits.
3. Record empirical failures and refine the rules instead of adding more
   speculative cases in advance.
4. Repeat until the segmentation and candidate-generation policy is stable,
   then freeze and version it.
5. Add MMLongBench query and gold-page navigation as a later viewer adapter.
6. Begin dataset creation only after the segmentation rules are finalized.

## Parallel planned gold-page baseline

The planned experiment compares one fixed answerer under two
gold-page-conditioned arms: all official gold pages at once versus the same
gold pages routed independently through the zero-shot reranker to quadrant
scale. Every gold page begins automatically positive; multiple positive
quadrants and cross-page branches are preserved; page/parent fallbacks remain
available; and fine-grained routing is deferred. No oracle quadrant is used.

The complete contract is
`docs/specifications/mmlongbench_gold_page_quadrant_reranker_experiment.md`.
The first prerequisite is the OCR-only SOL handoff at
`sol/task_spec/mmlongbench_gold_pages_deepseek_ocr2.md`.

## Next steps for a fresh local agent

1. Use the full hybrid review build to compare both strategies across the
   retained 306-page corpus.
2. Review more pages where visuals interrupt headed text, and record cases
   where the preserved baseline or hybrid produces a better routing unit.
3. Keep the baseline and hybrid separately versioned; do not replace the
   baseline until the manual comparison supports that decision.
4. Do not start query/gold-page integration as part of this comparison.
