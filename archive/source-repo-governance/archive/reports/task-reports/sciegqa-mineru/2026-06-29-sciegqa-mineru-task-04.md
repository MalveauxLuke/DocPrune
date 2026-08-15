# SciEGQA MinerU Pilot - Task 4 Evidence

## Scope

Task 4 reconstructs exact SciEGQA source pages from arXiv v1 PDFs. It pins the
PDF URL to v1, reads arXiv categories, infers page dimensions independently
from absolute and normalized annotation coordinates, renders one exact page at
300 DPI with Poppler, and rejects any pixel-dimension mismatch.

Implementation commit (immutable):
`6a48e315e66085664c109660abe05e2fb4b450b8`

## TDD Evidence

RED command:

```text
python -m pytest -q tests/test_sciegqa_mineru_source_pages.py
```

Observed expected failure before production code existed:

```text
ModuleNotFoundError: No module named 'scripts.sciegqa_mineru.source_pages'
1 error in 0.08s
```

GREEN focused command and result:

```text
python -m pytest -q tests/test_sciegqa_mineru_source_pages.py
............                                                             [100%]
12 passed in 0.08s
```

Full regression command and result:

```text
python -m pytest -q
........................................................................ [ 62%]
...........................................                              [100%]
115 passed in 2.19s
```

## Dependency Evidence

The local canonical renderer is available:

```text
pdftoppm version 26.03.0
Copyright 2005-2026 The Poppler Developers - http://poppler.freedesktop.org
```

The image inspection dependency imports successfully:

```text
Pillow 12.2.0
```

No production PDF was downloaded. Network requests and page creation were
tested at their standard-library boundaries; the real local Poppler binary and
Pillow import were checked directly.

## Files

- `scripts/sciegqa_mineru/source_pages.py`
- `tests/test_sciegqa_mineru_source_pages.py`
- `docs/superpowers/task-reports/2026-06-29-sciegqa-mineru-task-04.md`

## Contract and Error-Path Review

- The PDF URL is exactly `https://arxiv.org/pdf/{arxiv_id}v1`.
- Both arXiv requests send `sciegqa-mineru-pilot/1.0`; metadata uses a
  60-second timeout and PDF download uses a 120-second timeout.
- Downloads stream in 1 MiB reads and return URL, SHA-256, and byte count.
- Dimension inference requires at least one nonzero candidate per axis and
  rejects any candidate more than 0.1 pixels from the rounded mean.
- The Poppler command token order is exactly page-specific, 300 DPI,
  single-file PNG rendering with `check=True`.
- Rendered dimensions must equal the annotation-derived dimensions exactly;
  the implementation does not resize, clip, retry, or select another renderer.
- Atom responses without an entry and inconsistent or insufficient annotation
  geometry fail explicitly.

## Concerns and Deferred Runtime Evidence

There are no known implementation blockers. Live arXiv availability, an actual
v1 PDF download, and visual inspection of a production gold page are
deliberately deferred to subset materialization, where the selected document
and exact source-page number are available. This task introduces no fallback
that could hide those failures.
