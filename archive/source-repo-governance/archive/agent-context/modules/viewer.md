# MMLongBench Viewers

## Active applications

- `viewer/mmlongbench_ocr2/`: 306 successful pages from the unstructured OCR
  run, with atomic/semantic overlays and raw-output inspection.
- `viewer/mmlongbench_ocr2_deep_parse/`: retained 13-page synchronized
  comparison of eight prompt arms, 138 runs, and 47 visual crops.

## Active application

- `viewer/mmlongbench_segmentation_lab/`: local static manual-routing lab for
  the retained 306-page unstructured run. Build and serve commands are in its
  `README.md`; behavior is specified by
  [`docs/specifications/mmlongbench_segmentation_lab/design.md`](../../../docs/specifications/mmlongbench_segmentation_lab/design.md).
  It exposes the canonical `A/B/C` routing state machine under replaceable
  semantic-segmentation and boundary-extension strategies, with no SOL work.
  `deepseek_semantic_v1` is the preserved baseline;
  `deepseek_semantic_hybrid_v1` is the additive comparison that splits a
  semantic text section only when another DeepSeek unit interrupts its complete
  reading-order run. It creates no image-caption pairs.
- The lab workspace is viewport-bounded. Its page/routing divider changes pane
  width without rescaling the fixed-size page; each pane scrolls independently.
- The candidate panel has persistent **Live route** and **Visual route** tabs.
  Visual route projects the existing session candidates, expansions, and audit
  events into a selectable tree; selecting any node, including removed or
  reused candidates, highlights the same membership on the reference page
  without changing routing state. Deduplicated split children remain visible
  by their attempted kind and axis while pointing to the one canonical
  candidate rather than becoming duplicate scored candidates.
- Candidate cards, the reference summary, and the audit drawer display concise
  human candidate types and counts. Internal fingerprints, segment IDs, and
  routing-atom IDs remain available through JSON export but are not rendered
  in the manual-review workspace.

## Builders

- `scripts/mmlongbench/build_viewer.py`
- `scripts/mmlongbench/viewer/viewer_bundle.py`
- `scripts/mmlongbench/build_deep_parse_viewer.py`
- `scripts/mmlongbench/deep_parse_viewer/viewer_bundle.py`
- `scripts/mmlongbench/build_segmentation_lab.py`
- `scripts/mmlongbench/segmentation_lab/viewer_bundle.py`

## Tests

- `tests/test_mmlongbench_ocr2_viewer_bundle.py`
- `tests/test_mmlongbench_ocr2_viewer_static.py`
- `tests/test_mmlongbench_ocr2_deep_parse_viewer_bundle.py`
- `tests/test_mmlongbench_ocr2_deep_parse_viewer_static.py`
- `tests/test_mmlongbench_segmentation_lab_integration.py`

Serve the retained deep-parse bundle:

```bash
python -m http.server 8010 --directory viewer/mmlongbench_ocr2_deep_parse
```

Do not treat generated viewer pages or manifests as source unless the retained
experiment explicitly tracks them.
