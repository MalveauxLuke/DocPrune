# SciEGQA MinerU Task 4 - Multi-Dot Page Name Fix

## Scope and Immutable Implementation

This correction preserves multi-dot page stems when locating Poppler's staged
PNG. It does not change suffix validation, rendering options, image validation,
atomic publication, metadata, or failure behavior.

Implementation commit:
`0d5c54cc398be6ea57e4ee728d35a6159be1eb68`

## Root Cause and RED Evidence

For canonical `page.v1.png`, the staged Poppler prefix is `page.v1`. Poppler
appends `.png` and emits `page.v1.png`, while `Path.with_suffix(".png")`
incorrectly changed the expected staged path to `page.png`.

The real-Poppler reproduction before the fix reported:

```text
FileNotFoundError: [Errno 2] No such file or directory: '.../page.png'
['synthetic.pdf']
```

The unit regression then failed at the same boundary:

```text
python -m pytest -q tests/test_sciegqa_mineru_source_pages.py
FAILED test_renders_multi_dot_page_name_without_losing_stem
1 failed, 28 passed, 17 subtests passed in 0.16s
```

## Fix

The staged PNG path now uses literal extension append:

```python
staged_page = Path(f"{output_prefix}.png")
```

This matches Poppler's output contract for both simple and multi-dot prefixes.

## GREEN and Regression Evidence

Focused suite:

```text
python -m pytest -q tests/test_sciegqa_mineru_source_pages.py
.............................                           [100%]
29 passed, 17 subtests passed in 0.11s
```

Full suite:

```text
python -m pytest -q
........................................................................ [ 54%]
....................................................... [ 96%]
.....                                                                    [100%]
132 passed, 17 subtests passed in 2.08s
```

`git diff --check` exited successfully.

## Independent Real-Poppler Multi-Dot Probe

A synthetic one-page PDF was rendered to `page.v1.png` with the real local
Poppler binary. The successful artifact's hash matched returned metadata, and
the output directory contained only the synthetic PDF and canonical PNG.

```text
multi_dot_success= {'width_px': 300, 'height_px': 300, 'render_dpi': 300,
  'image_sha256': '48279a161aa7fa35436aa21f1334e08df934ccb6e50a57e91046e129da1eab0f'}
multi_dot_failure= rendered page size (300, 300) does not match annotation-derived (301, 300)
canonical_preserved=True, temp_residue=False
```

The intentional wrong-dimension render confirmed the existing canonical PNG
remained byte-identical and the temporary staging directory was removed.

No production document was downloaded or rendered.
