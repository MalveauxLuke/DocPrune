# SciEGQA-Train MinerU Evidence-Page Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, category-stratified 20-question SciEGQA-Train SPSR subset, parse its validated evidence pages with a local MinerU 3.4 high-effort service, and inspect exact gold and MinerU geometry in a static page-first web viewer.

**Architecture:** Keep preprocessing and inspection separate. Python modules select and validate source records, reconstruct arXiv v1 pages, invoke a persistent localhost MinerU service, and canonicalize immutable artifacts into entity JSONL files; a static viewer consumes a derived manifest without starting or modifying MinerU. Generated PDFs, page images, model outputs, and the built site stay under the already ignored `outputs/` tree.

**Tech Stack:** Python 3.12, standard library, `huggingface_hub`, `pypdf`, Poppler `pdftoppm`, MinerU 3.4.0, pytest/unittest, vanilla HTML/CSS/JavaScript, SVG overlays.

---

## File Map

### Environment and operation

- Create `environments/mineru34-macos.yml` — isolated local MinerU environment.
- Create `scripts/run_mineru_local_api.sh` — localhost-only, single-request MinerU API launcher.
- Create `docs/mineru_local_setup.md` — exact setup, smoke-test, run, and inspection commands.
- Modify `.gitignore` — ignore `.superpowers/` visual-companion artifacts.

### Pipeline package

- Create `scripts/sciegqa_mineru/__init__.py` — package marker and schema version.
- Create `scripts/sciegqa_mineru/schema.py` — stable IDs, hashes, JSONL helpers, and bbox validation/conversion.
- Create `scripts/sciegqa_mineru/selection.py` — SPSR filtering and deterministic category-quota selection.
- Create `scripts/sciegqa_mineru/source_pages.py` — arXiv v1 download, 300-DPI render, and dimension validation.
- Create `scripts/sciegqa_mineru/mineru_runner.py` — MinerU CLI/API orchestration and raw artifact inventory.
- Create `scripts/sciegqa_mineru/canonicalize.py` — MinerU content-list validation and segment records.
- Create `scripts/sciegqa_mineru/viewer_bundle.py` — self-contained viewer manifest and asset bundle.

### Command-line entry points

- Create `scripts/build_sciegqa_mineru_subset.py` — dataset resolution, selection, source materialization, and provenance output.
- Create `scripts/run_sciegqa_mineru.py` — one-page smoke test and full unique-page parsing.
- Create `scripts/build_sciegqa_mineru_viewer.py` — canonical segment and static-site build.

### Viewer

- Create `viewer/sciegqa_mineru/index.html` — page-first inspector structure.
- Create `viewer/sciegqa_mineru/styles.css` — responsive page canvas, side panel, legend, and layer styles.
- Create `viewer/sciegqa_mineru/app.js` — navigation, mode switching, SVG geometry, filters, and segment details.

### Tests

- Create `tests/test_sciegqa_mineru_schema.py`
- Create `tests/test_sciegqa_mineru_selection.py`
- Create `tests/test_sciegqa_mineru_source_pages.py`
- Create `tests/test_sciegqa_mineru_runner.py`
- Create `tests/test_sciegqa_mineru_canonicalize.py`
- Create `tests/test_sciegqa_mineru_viewer_bundle.py`
- Create `tests/test_sciegqa_mineru_viewer_static.py`
- Create `tests/test_mineru_local_setup.py`
- Create `tests/test_sciegqa_mineru_end_to_end.py`

## Task 1: Isolated MinerU Environment and Local Service

**Files:**
- Create: `environments/mineru34-macos.yml`
- Create: `scripts/run_mineru_local_api.sh`
- Create: `tests/test_mineru_local_setup.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write the failing environment and launcher contract test**

```python
# tests/test_mineru_local_setup.py
import unittest
from pathlib import Path


class MinerULocalSetupTests(unittest.TestCase):
    def test_environment_pins_python_and_mineru(self):
        text = Path("environments/mineru34-macos.yml").read_text(encoding="utf-8")
        self.assertIn("python=3.12", text)
        self.assertIn('mineru[all]==3.4.0', text)

    def test_launcher_is_local_and_memory_bounded(self):
        text = Path("scripts/run_mineru_local_api.sh").read_text(encoding="utf-8")
        self.assertIn("--host 127.0.0.1", text)
        self.assertIn("MINERU_API_MAX_CONCURRENT_REQUESTS", text)
        self.assertIn("MINERU_PROCESSING_WINDOW_SIZE", text)
        self.assertIn("--enable-vlm-preload true", text)
        self.assertNotIn("0.0.0.0", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify it fails because files are absent**

Run: `pytest -q tests/test_mineru_local_setup.py`

Expected: FAIL with `FileNotFoundError` for `environments/mineru34-macos.yml`.

- [ ] **Step 3: Add the pinned environment**

```yaml
# environments/mineru34-macos.yml
name: mineru34
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - pip:
      - "mineru[all]==3.4.0"
      - pytest
```

- [ ] **Step 4: Add the local API launcher**

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
ENV_PREFIX="${MINERU_ENV_PREFIX:-$HOME/miniforge3/envs/mineru34}"
API_PORT="${MINERU_API_PORT:-8000}"
API_OUTPUT_DIR="${MINERU_API_OUTPUT_DIR:-$PROJECT_DIR/outputs/sciegqa_mineru_pilot/mineru_api}"

mkdir -p "$API_OUTPUT_DIR"
cd "$API_OUTPUT_DIR"

export MINERU_MODEL_SOURCE="${MINERU_MODEL_SOURCE:-huggingface}"
export MINERU_API_MAX_CONCURRENT_REQUESTS="${MINERU_API_MAX_CONCURRENT_REQUESTS:-1}"
export MINERU_PROCESSING_WINDOW_SIZE="${MINERU_PROCESSING_WINDOW_SIZE:-1}"
export MINERU_PDF_RENDER_THREADS="${MINERU_PDF_RENDER_THREADS:-2}"

exec "$ENV_PREFIX/bin/mineru-api" \
  --host 127.0.0.1 \
  --port "$API_PORT" \
  --enable-vlm-preload true
```

Run: `chmod +x scripts/run_mineru_local_api.sh`

- [ ] **Step 5: Ignore visual-companion scratch files**

Add only this line to `.gitignore`:

```gitignore
.superpowers/
```

- [ ] **Step 6: Run the focused test**

Run: `pytest -q tests/test_mineru_local_setup.py`

Expected: `2 passed`.

- [ ] **Step 7: Create the environment and verify the pinned CLI**

Run:

```bash
mamba env create -f environments/mineru34-macos.yml
$HOME/miniforge3/envs/mineru34/bin/mineru --version
```

Expected: the second command prints `3.4.0`.

- [ ] **Step 8: Commit the isolated environment and launcher**

Before committing, run `git branch --show-current` and verify `COLQWEN_binary_classification`.

```bash
git add .gitignore environments/mineru34-macos.yml scripts/run_mineru_local_api.sh tests/test_mineru_local_setup.py
git commit -m "build: add isolated MinerU 3.4 environment"
```

## Task 2: Canonical Schema and Geometry Contract

**Files:**
- Create: `scripts/sciegqa_mineru/__init__.py`
- Create: `scripts/sciegqa_mineru/schema.py`
- Create: `tests/test_sciegqa_mineru_schema.py`

- [ ] **Step 1: Write failing geometry and stable-ID tests**

```python
# tests/test_sciegqa_mineru_schema.py
import json
import tempfile
import unittest
from pathlib import Path

from scripts.sciegqa_mineru.schema import (
    BBox,
    artifact_sha256,
    make_stable_id,
    read_jsonl,
    write_jsonl,
)


class SchemaTests(unittest.TestCase):
    def test_bbox_round_trip_preserves_coordinates(self):
        box = BBox(528.012898, 77.251995, 873.43813, 320.980616)
        pixels = box.norm1000_to_pixels(width=2481, height=3508)
        self.assertAlmostEqual(pixels.x0, 1310.0, places=4)
        self.assertAlmostEqual(pixels.y0, 271.0, places=4)
        round_trip = pixels.pixels_to_norm1000(width=2481, height=3508)
        for actual, expected in zip(round_trip.as_list(), box.as_list()):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_invalid_bbox_is_rejected_without_clipping(self):
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            BBox(-1, 0, 100, 100).validate(bounds=(1000, 1000))

    def test_stable_id_uses_source_identity(self):
        first = make_stable_id("query", "Yuwh07/SciEGQA-Train", "abc123", 42)
        second = make_stable_id("query", "Yuwh07/SciEGQA-Train", "abc123", 42)
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("query_"))

    def test_jsonl_and_hash_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            write_jsonl(path, [{"a": 1}, {"a": 2}])
            self.assertEqual(read_jsonl(path), [{"a": 1}, {"a": 2}])
            self.assertEqual(len(artifact_sha256(path)), 64)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify import failure**

Run: `pytest -q tests/test_sciegqa_mineru_schema.py`

Expected: collection ERROR with `ModuleNotFoundError: scripts.sciegqa_mineru`.

- [ ] **Step 3: Add package metadata**

```python
# scripts/sciegqa_mineru/__init__.py
SCHEMA_VERSION = "1.0"
```

- [ ] **Step 4: Implement strict bbox, ID, hash, and JSONL helpers**

```python
# scripts/sciegqa_mineru/schema.py
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_list(cls, values: list[float]) -> "BBox":
        if len(values) != 4:
            raise ValueError(f"bbox must have four values, got {values!r}")
        return cls(*(float(value) for value in values))

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    def validate(self, bounds: tuple[float, float]) -> "BBox":
        if not all(math.isfinite(value) for value in self.as_list()):
            raise ValueError(f"bbox contains non-finite value: {self.as_list()}")
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError(f"bbox coordinates are not ordered: {self.as_list()}")
        width, height = bounds
        if self.x0 < 0 or self.y0 < 0 or self.x1 > width or self.y1 > height:
            raise ValueError(f"bbox is out of bounds {bounds}: {self.as_list()}")
        return self

    def norm1000_to_pixels(self, width: int, height: int) -> "BBox":
        self.validate((1000, 1000))
        return BBox(
            self.x0 * width / 1000,
            self.y0 * height / 1000,
            self.x1 * width / 1000,
            self.y1 * height / 1000,
        )

    def pixels_to_norm1000(self, width: int, height: int) -> "BBox":
        self.validate((width, height))
        return BBox(
            self.x0 * 1000 / width,
            self.y0 * 1000 / height,
            self.x1 * 1000 / width,
            self.y1 * 1000 / height,
        )


def make_stable_id(kind: str, *parts: object) -> str:
    identity = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"{kind}_{digest}"


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
```

- [ ] **Step 5: Run the focused tests**

Run: `pytest -q tests/test_sciegqa_mineru_schema.py`

Expected: `4 passed`.

- [ ] **Step 6: Commit the canonical schema**

Before committing, run `git branch --show-current`.

```bash
git add scripts/sciegqa_mineru/__init__.py scripts/sciegqa_mineru/schema.py tests/test_sciegqa_mineru_schema.py
git commit -m "feat: define SciEGQA MinerU provenance schema"
```

## Task 3: Deterministic SPSR Selection

**Files:**
- Create: `scripts/sciegqa_mineru/selection.py`
- Create: `tests/test_sciegqa_mineru_selection.py`

- [ ] **Step 1: Write failing eligibility, quota, and document-isolation tests**

```python
# tests/test_sciegqa_mineru_selection.py
import unittest

from scripts.sciegqa_mineru.selection import (
    CATEGORY_QUOTAS,
    is_spsr_row,
    select_rows,
)


def row(index, category, doc, page=1):
    return {
        "source_record_index": index,
        "query": f"q{index}",
        "answer": f"a{index}",
        "doc_name": doc,
        "category": category,
        "evidence_page": [page],
        "bbox": [[[10, 20, 100, 200]]],
        "rel_bbox": [[[10, 20, 100, 200]]],
        "subimg_type": [["image"]],
    }


class SelectionTests(unittest.TestCase):
    def test_spsr_requires_one_page_and_one_box(self):
        self.assertTrue(is_spsr_row(row(1, "cs", "doc-a")))
        invalid = row(2, "cs", "doc-b")
        invalid["bbox"] = [[[1, 2, 3, 4], [5, 6, 7, 8]]]
        self.assertFalse(is_spsr_row(invalid))

    def test_selects_exact_quotas_without_reusing_documents(self):
        quotas = {"q-fin": 2, "econ": 1, "math": 1}
        rows = [
            row(0, "q-fin", "cross"),
            row(1, "q-fin", "cross"),
            row(2, "econ", "cross"),
            row(3, "econ", "econ-only"),
            row(4, "math", "math-only"),
            row(5, "q-fin", "qfin-alt"),
            row(6, "q-fin", "qfin-alt"),
        ]
        selected, audit = select_rows(rows, quotas)
        self.assertEqual([item["source_record_index"] for item in selected], [0, 1, 3, 4])
        self.assertEqual({item["doc_name"] for item in selected}, {"cross", "econ-only", "math-only"})
        self.assertEqual(audit["category_counts"], quotas)

        replacement, _ = select_rows(rows, quotas, excluded_docs={"cross"})
        self.assertEqual([item["source_record_index"] for item in replacement], [5, 6, 3, 4])

    def test_production_quotas_total_twenty(self):
        self.assertEqual(sum(CATEGORY_QUOTAS.values()), 20)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify import failure**

Run: `pytest -q tests/test_sciegqa_mineru_selection.py`

Expected: collection ERROR because `selection.py` does not exist.

- [ ] **Step 3: Implement strict SPSR filtering and deterministic selection**

```python
# scripts/sciegqa_mineru/selection.py
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from scripts.sciegqa_mineru.schema import BBox


CATEGORY_QUOTAS = {
    "q-fin": 4,
    "q-bio": 3,
    "eess": 3,
    "physics": 3,
    "cs": 2,
    "econ": 2,
    "stat": 2,
    "math": 1,
}


def is_spsr_row(row: dict[str, Any]) -> bool:
    try:
        if len(row["evidence_page"]) != 1:
            return False
        if len(row["bbox"]) != 1 or len(row["bbox"][0]) != 1:
            return False
        if len(row["rel_bbox"]) != 1 or len(row["rel_bbox"][0]) != 1:
            return False
        BBox.from_list(row["rel_bbox"][0][0]).validate((1000, 1000))
        pixel_box = BBox.from_list(row["bbox"][0][0])
        if not (pixel_box.x0 < pixel_box.x1 and pixel_box.y0 < pixel_box.y1):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def select_rows(
    rows: list[dict[str, Any]],
    quotas: dict[str, int] = CATEGORY_QUOTAS,
    excluded_docs: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded_docs = excluded_docs or set()
    eligible = [row for row in rows if is_spsr_row(row)]
    by_category_doc: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for item in eligible:
        by_category_doc[str(item["category"])][str(item["doc_name"])].append(item)

    selected: list[dict[str, Any]] = []
    assigned_docs: set[str] = set()
    chosen_docs: dict[str, str] = {}
    for category, quota in quotas.items():
        candidates = sorted(
            by_category_doc[category].items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
        for doc_name, doc_rows in candidates:
            if doc_name in excluded_docs or doc_name in assigned_docs or len(doc_rows) < quota:
                continue
            ordered = sorted(doc_rows, key=lambda item: int(item["source_record_index"]))
            selected.extend(ordered[:quota])
            assigned_docs.add(doc_name)
            chosen_docs[category] = doc_name
            break
        else:
            raise ValueError(f"No unused document can satisfy {category} quota {quota}")

    counts = Counter(str(item["category"]) for item in selected)
    audit = {
        "selection_strategy": "spsr_category_quota_high_yield_document",
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "category_counts": dict(counts),
        "chosen_documents": chosen_docs,
    }
    return selected, audit
```

- [ ] **Step 4: Run the focused tests**

Run: `pytest -q tests/test_sciegqa_mineru_selection.py`

Expected: `3 passed`.

- [ ] **Step 5: Commit deterministic selection**

Before committing, run `git branch --show-current`.

```bash
git add scripts/sciegqa_mineru/selection.py tests/test_sciegqa_mineru_selection.py
git commit -m "feat: select stratified SciEGQA SPSR rows"
```

## Task 4: arXiv v1 Source Pages and Dimension Validation

**Files:**
- Create: `scripts/sciegqa_mineru/source_pages.py`
- Create: `tests/test_sciegqa_mineru_source_pages.py`

- [ ] **Step 1: Write failing dimension and command-construction tests**

```python
# tests/test_sciegqa_mineru_source_pages.py
import unittest

from scripts.sciegqa_mineru.schema import BBox
from scripts.sciegqa_mineru.source_pages import (
    arxiv_v1_pdf_url,
    infer_page_dimension,
    render_command,
)


class SourcePageTests(unittest.TestCase):
    def test_arxiv_url_is_pinned_to_v1(self):
        self.assertEqual(arxiv_v1_pdf_url("2411.02804"), "https://arxiv.org/pdf/2411.02804v1")

    def test_infers_exact_page_dimensions(self):
        pixel = BBox(1310, 271, 2167, 1126)
        norm = BBox(528.012898, 77.251995, 873.43813, 320.980616)
        self.assertEqual(infer_page_dimension(pixel, norm), (2481, 3508))

    def test_render_command_is_page_specific_and_300_dpi(self):
        command = render_command("paper.pdf", "page", source_page_number=7)
        self.assertEqual(
            command,
            ["pdftoppm", "-f", "7", "-l", "7", "-r", "300", "-singlefile", "-png", "paper.pdf", "page"],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify import failure**

Run: `pytest -q tests/test_sciegqa_mineru_source_pages.py`

Expected: collection ERROR because `source_pages.py` does not exist.

- [ ] **Step 3: Implement v1 download, render command, and strict dimension inference**

```python
# scripts/sciegqa_mineru/source_pages.py
from __future__ import annotations

import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from scripts.sciegqa_mineru.schema import BBox, artifact_sha256


class SourceDocumentError(RuntimeError):
    def __init__(self, doc_name: str, message: str):
        super().__init__(message)
        self.doc_name = doc_name


def arxiv_v1_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}v1"


def parse_arxiv_categories(atom_xml: bytes) -> list[str]:
    root = ET.fromstring(atom_xml)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", namespace)
    if entry is None:
        raise ValueError("arXiv metadata response has no entry")
    return sorted({node.attrib["term"] for node in entry.findall("atom:category", namespace)})


def fetch_arxiv_categories(arxiv_id: str) -> list[str]:
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    request = urllib.request.Request(url, headers={"User-Agent": "sciegqa-mineru-pilot/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return parse_arxiv_categories(response.read())


def infer_page_dimension(pixel: BBox, norm: BBox) -> tuple[int, int]:
    width_candidates = [
        px * 1000 / rel
        for px, rel in ((pixel.x0, norm.x0), (pixel.x1, norm.x1))
        if px != 0 and rel != 0
    ]
    height_candidates = [
        px * 1000 / rel
        for px, rel in ((pixel.y0, norm.y0), (pixel.y1, norm.y1))
        if px != 0 and rel != 0
    ]
    if not width_candidates or not height_candidates:
        raise ValueError("bbox lacks nonzero coordinates for dimension inference")
    def resolve(candidates: list[float], axis: str) -> int:
        result = round(sum(candidates) / len(candidates))
        if any(abs(value - result) > 0.1 for value in candidates):
            raise ValueError(f"annotation-derived {axis} is inconsistent: {candidates}")
        return result

    return resolve(width_candidates, "width"), resolve(height_candidates, "height")


def render_command(pdf: str, output_prefix: str, source_page_number: int) -> list[str]:
    page = str(source_page_number)
    return ["pdftoppm", "-f", page, "-l", page, "-r", "300", "-singlefile", "-png", pdf, output_prefix]


def download_pdf(arxiv_id: str, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = arxiv_v1_pdf_url(arxiv_id)
    request = urllib.request.Request(url, headers={"User-Agent": "sciegqa-mineru-pilot/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    return {
        "pdf_url": url,
        "pdf_sha256": artifact_sha256(destination),
        "pdf_bytes": destination.stat().st_size,
    }


def render_and_validate_page(
    pdf_path: Path,
    page_path: Path,
    source_page_number: int,
    expected_size: tuple[int, int],
) -> dict[str, object]:
    page_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = page_path.with_suffix("")
    subprocess.run(render_command(str(pdf_path), str(prefix), source_page_number), check=True)
    with Image.open(page_path) as image:
        size = image.size
    if size != expected_size:
        raise ValueError(f"rendered page size {size} does not match annotation-derived {expected_size}")
    return {
        "width_px": size[0],
        "height_px": size[1],
        "render_dpi": 300,
        "image_sha256": artifact_sha256(page_path),
    }
```

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_sciegqa_mineru_source_pages.py`

Expected: `3 passed`.

- [ ] **Step 5: Commit source-page reconstruction**

Before committing, run `git branch --show-current`.

```bash
git add scripts/sciegqa_mineru/source_pages.py tests/test_sciegqa_mineru_source_pages.py
git commit -m "feat: reconstruct validated SciEGQA source pages"
```

## Task 5: Build the Provenance-Preserving Subset

**Files:**
- Create: `scripts/build_sciegqa_mineru_subset.py`
- Create: `tests/test_sciegqa_mineru_end_to_end.py`

- [ ] **Step 1: Write a failing small-fixture subset test**

```python
# tests/test_sciegqa_mineru_end_to_end.py
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_sciegqa_mineru_subset import build_subset


class SubsetBuildTests(unittest.TestCase):
    def test_build_writes_joinable_entities_and_selection_audit(self):
        rows = []
        quotas = {"cs": 1, "math": 1}
        for index, category in enumerate(quotas):
            rows.append({
                "query": f"question-{index}",
                "answer": f"answer-{index}",
                "doc_name": f"2401.0000{index}",
                "category": category,
                "evidence_page": [1],
                "bbox": [[[255, 330, 510, 660]]],
                "rel_bbox": [[[100, 100, 200, 200]]],
                "subimg_type": [["image"]],
            })
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "train.jsonl"
            dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with patch("scripts.build_sciegqa_mineru_subset.materialize_document_pages") as materialize:
                materialize.side_effect = lambda **kwargs: kwargs["entity_records"]
                summary = build_subset(dataset, "revision-sha", root / "out", quotas)
            self.assertEqual(summary["query_count"], 2)
            self.assertEqual(len((root / "out" / "queries.jsonl").read_text().splitlines()), 2)
            self.assertEqual(len((root / "out" / "evidence.jsonl").read_text().splitlines()), 2)
            self.assertTrue((root / "out" / "selection_audit.json").is_file())
            self.assertTrue((root / "out" / "dataset_source.json").is_file())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify import failure**

Run: `pytest -q tests/test_sciegqa_mineru_end_to_end.py`

Expected: collection ERROR because `build_sciegqa_mineru_subset.py` does not exist.

- [ ] **Step 3: Implement the CLI orchestration boundary**

The new script must expose these exact functions so network and rendering are mockable:

```python
DATASET_REPO = "Yuwh07/SciEGQA-Train"
DATASET_FILENAME = "SciEGQA-Train.jsonl"

def resolve_dataset(revision: str) -> tuple[Path, str]:
    info = HfApi().dataset_info(DATASET_REPO, revision=revision)
    path = Path(hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename=DATASET_FILENAME,
        revision=info.sha,
    ))
    return path, info.sha

def build_subset(
    dataset_path: Path,
    dataset_revision: str,
    output_dir: Path,
    quotas: dict[str, int] = CATEGORY_QUOTAS,
) -> dict[str, object]:
    raw_rows = read_jsonl(dataset_path)
    indexed = [dict(row, source_record_index=index) for index, row in enumerate(raw_rows)]
    excluded_docs: set[str] = set()
    source_rejections: list[dict[str, str]] = []
    while True:
        selected, audit = select_rows(indexed, quotas, excluded_docs=excluded_docs)
        entity_records = build_entity_records(selected, dataset_revision)
        try:
            materialized = materialize_document_pages(
                output_dir=output_dir,
                entity_records=entity_records,
            )
        except SourceDocumentError as exc:
            excluded_docs.add(exc.doc_name)
            source_rejections.append({"doc_name": exc.doc_name, "reason": str(exc)})
            continue
        break
    audit["source_rejections"] = source_rejections
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_source.json").write_text(json.dumps({
        "dataset_name": DATASET_REPO,
        "dataset_filename": DATASET_FILENAME,
        "dataset_revision": dataset_revision,
        "dataset_jsonl_sha256": artifact_sha256(dataset_path),
    }, indent=2, sort_keys=True), encoding="utf-8")
    write_jsonl(output_dir / "documents.jsonl", materialized["documents"])
    write_jsonl(output_dir / "pages.jsonl", materialized["pages"])
    write_jsonl(output_dir / "queries.jsonl", materialized["queries"])
    write_jsonl(output_dir / "evidence.jsonl", materialized["evidence"])
    (output_dir / "selection_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "query_count": len(materialized["queries"]),
        "document_count": len(materialized["documents"]),
        "page_count": len(materialized["pages"]),
        "dataset_revision": dataset_revision,
    }
```

`build_entity_records()` must use `make_stable_id()` for every entity, preserve both source box representations, set `annotation_origin` to `sciegqa_train_automatic`, and keep source page number and index separately. `materialize_document_pages()` must group selected rows by document/page, call Task 4 helpers, and return the four entity lists without fuzzy matching or fallback resizing.

Use these exact entity-building boundaries:

```python
def build_entity_records(selected: list[dict], dataset_revision: str) -> dict[str, list[dict]]:
    documents: dict[str, dict] = {}
    pages: dict[str, dict] = {}
    queries: list[dict] = []
    evidence: list[dict] = []
    for row in selected:
        doc_name = str(row["doc_name"])
        page_number = int(row["evidence_page"][0])
        document_id = make_stable_id("document", DATASET_REPO, dataset_revision, doc_name)
        page_id = make_stable_id("page", document_id, page_number)
        query_id = make_stable_id(
            "query", DATASET_REPO, dataset_revision, int(row["source_record_index"])
        )
        evidence_id = make_stable_id("evidence", query_id, page_id, 0)
        pixel_box = BBox.from_list(row["bbox"][0][0])
        norm_box = BBox.from_list(row["rel_bbox"][0][0]).validate((1000, 1000))
        expected_width, expected_height = infer_page_dimension(pixel_box, norm_box)
        documents.setdefault(document_id, {
            "document_id": document_id,
            "arxiv_id": doc_name,
            "arxiv_version": "v1",
            "pdf_url": arxiv_v1_pdf_url(doc_name),
        })
        pages.setdefault(page_id, {
            "page_id": page_id,
            "document_id": document_id,
            "source_page_number": page_number,
            "source_page_index": page_number - 1,
            "expected_width_px": expected_width,
            "expected_height_px": expected_height,
        })
        queries.append({
            "query_id": query_id,
            "dataset_name": DATASET_REPO,
            "dataset_revision": dataset_revision,
            "source_record_index": int(row["source_record_index"]),
            "source_category": str(row["category"]),
            "question": str(row["query"]),
            "answer": str(row["answer"]),
            "page_id": page_id,
            "selection_rule": "spsr_category_quota_high_yield_document",
        })
        evidence.append({
            "evidence_id": evidence_id,
            "query_id": query_id,
            "page_id": page_id,
            "annotation_origin": "sciegqa_train_automatic",
            "source_bbox_px": pixel_box.as_list(),
            "source_bbox_norm_1000": norm_box.as_list(),
            "source_subimg_type": str(row["subimg_type"][0][0]),
        })
    return {
        "documents": sorted(documents.values(), key=lambda item: item["arxiv_id"]),
        "pages": sorted(pages.values(), key=lambda item: (item["document_id"], item["source_page_number"])),
        "queries": sorted(queries, key=lambda item: item["source_record_index"]),
        "evidence": sorted(evidence, key=lambda item: item["query_id"]),
    }


def materialize_document_pages(output_dir: Path, entity_records: dict[str, list[dict]]) -> dict[str, list[dict]]:
    documents = {row["document_id"]: dict(row) for row in entity_records["documents"]}
    pages_by_document: dict[str, list[dict]] = defaultdict(list)
    for row in entity_records["pages"]:
        pages_by_document[row["document_id"]].append(dict(row))
    materialized_pages: list[dict] = []
    for document_id, page_rows in sorted(pages_by_document.items()):
        document = documents[document_id]
        arxiv_id = document["arxiv_id"]
        doc_dir = output_dir / "source" / arxiv_id
        pdf_path = doc_dir / f"{arxiv_id}v1.pdf"
        try:
            pdf_meta = download_pdf(arxiv_id, pdf_path)
            document.update(pdf_meta)
            document["pdf_path"] = str(pdf_path.relative_to(output_dir))
            document["arxiv_categories"] = fetch_arxiv_categories(arxiv_id)
            document["page_count"] = len(PdfReader(pdf_path).pages)
            for page in sorted(page_rows, key=lambda item: item["source_page_number"]):
                page_path = doc_dir / "pages" / f"page_{page['source_page_number']:05d}.png"
                page_meta = render_and_validate_page(
                    pdf_path,
                    page_path,
                    page["source_page_number"],
                    (page["expected_width_px"], page["expected_height_px"]),
                )
                page.update(page_meta)
                page["image_path"] = str(page_path.relative_to(output_dir))
                materialized_pages.append(page)
        except Exception as exc:
            raise SourceDocumentError(arxiv_id, f"{arxiv_id} source validation failed: {exc}") from exc
    return {
        "documents": sorted(documents.values(), key=lambda item: item["arxiv_id"]),
        "pages": materialized_pages,
        "queries": entity_records["queries"],
        "evidence": entity_records["evidence"],
    }
```

- [ ] **Step 4: Add argument parsing and exact expected output**

```python
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dataset-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sciegqa_mineru_pilot"))
    args = parser.parse_args()
    if args.dataset_jsonl:
        dataset_path, revision = args.dataset_jsonl, args.revision
    else:
        dataset_path, revision = resolve_dataset(args.revision)
    print(json.dumps(build_subset(dataset_path, revision, args.output_dir), indent=2))
    return 0
```

- [ ] **Step 5: Run focused tests**

Run: `pytest -q tests/test_sciegqa_mineru_end_to_end.py tests/test_sciegqa_mineru_selection.py tests/test_sciegqa_mineru_source_pages.py`

Expected: all tests PASS.

- [ ] **Step 6: Run the real selection and source-page build**

Run:

```bash
$HOME/miniforge3/envs/mineru34/bin/python scripts/build_sciegqa_mineru_subset.py \
  --output-dir outputs/sciegqa_mineru_pilot
```

Expected: JSON summary with `query_count: 20`, the eight approved category counts in `selection_audit.json`, and no dimension-validation error.

- [ ] **Step 7: Inspect selection before parsing**

Run:

```bash
wc -l outputs/sciegqa_mineru_pilot/{documents,pages,queries,evidence}.jsonl
jq '.category_counts, .chosen_documents' outputs/sciegqa_mineru_pilot/selection_audit.json
```

Expected: 20 query and evidence rows, no repeated chosen document, and exact quota counts.

- [ ] **Step 8: Commit subset construction code and tests**

Do not stage generated `outputs/`. Before committing, run `git branch --show-current`.

```bash
git add scripts/build_sciegqa_mineru_subset.py tests/test_sciegqa_mineru_end_to_end.py
git commit -m "feat: build SciEGQA MinerU pilot subset"
```

## Task 6: MinerU Page Runner and One-Page Smoke Gate

**Files:**
- Create: `scripts/sciegqa_mineru/mineru_runner.py`
- Create: `scripts/run_sciegqa_mineru.py`
- Create: `tests/test_sciegqa_mineru_runner.py`

- [ ] **Step 1: Write failing command and artifact-inventory tests**

```python
# tests/test_sciegqa_mineru_runner.py
import json
import tempfile
import unittest
from pathlib import Path

from scripts.sciegqa_mineru.mineru_runner import build_mineru_command, inventory_artifacts


class MinerURunnerTests(unittest.TestCase):
    def test_command_uses_high_effort_and_exact_page(self):
        command = build_mineru_command(
            mineru_bin=Path("/env/bin/mineru"),
            pdf_path=Path("paper.pdf"),
            output_dir=Path("out"),
            api_url="http://127.0.0.1:8000",
            source_page_index=6,
        )
        self.assertIn("hybrid-engine", command)
        self.assertEqual(command[command.index("--effort") + 1], "high")
        self.assertEqual(command[command.index("--image-analysis") + 1], "true")
        self.assertEqual(command[command.index("--start") + 1], "6")
        self.assertEqual(command[command.index("--end") + 1], "6")

    def test_inventory_requires_each_raw_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for suffix, content in {
                "_content_list.json": "[]",
                "_middle.json": "{}",
                "_model.json": "[]",
            }.items():
                (root / f"paper{suffix}").write_text(content, encoding="utf-8")
            (root / "paper_layout.pdf").write_bytes(b"%PDF")
            inventory = inventory_artifacts(root)
            self.assertEqual(set(inventory), {"content_list", "middle", "model", "layout"})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `pytest -q tests/test_sciegqa_mineru_runner.py`

Expected: collection ERROR because `mineru_runner.py` does not exist.

- [ ] **Step 3: Implement exact command construction and artifact enforcement**

```python
# scripts/sciegqa_mineru/mineru_runner.py
from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.sciegqa_mineru.schema import artifact_sha256


REQUIRED_PATTERNS = {
    "content_list": "*_content_list.json",
    "middle": "*_middle.json",
    "model": "*_model.json",
    "layout": "*_layout.pdf",
}


def build_mineru_command(
    mineru_bin: Path,
    pdf_path: Path,
    output_dir: Path,
    api_url: str,
    source_page_index: int,
) -> list[str]:
    page = str(source_page_index)
    return [
        str(mineru_bin), "--path", str(pdf_path), "--output", str(output_dir),
        "--api-url", api_url, "--backend", "hybrid-engine", "--effort", "high",
        "--image-analysis", "true", "--formula", "true", "--table", "true",
        "--start", page, "--end", page,
    ]


def inventory_artifacts(output_dir: Path) -> dict[str, dict[str, str]]:
    inventory = {}
    for name, pattern in REQUIRED_PATTERNS.items():
        matches = sorted(output_dir.rglob(pattern))
        if len(matches) != 1:
            raise ValueError(f"expected one {name} artifact under {output_dir}, found {len(matches)}")
        inventory[name] = {"path": str(matches[0]), "sha256": artifact_sha256(matches[0])}
    return inventory


def run_page(command: list[str], output_dir: Path) -> dict[str, dict[str, str]]:
    subprocess.run(command, check=True)
    return inventory_artifacts(output_dir)
```

- [ ] **Step 4: Implement page deduplication, smoke mode, runtime manifest, and full mode**

`scripts/run_sciegqa_mineru.py` must:

1. load `pages.jsonl` and `documents.jsonl`;
2. verify `GET /health` succeeds before parsing;
3. choose the first page only when `--smoke` is set;
4. create one output directory per stable `page_id`;
5. invoke `run_page()` once per unique page;
6. write `mineru_runs.jsonl` with page ID, command, package version, model cache revision, parameters, timestamps, status, and artifact hashes; and
7. stop immediately on the first failed page without marking later pages complete.

Use this orchestration shape:

```python
def api_health(api_url: str) -> dict:
    with urllib.request.urlopen(f"{api_url.rstrip('/')}/health", timeout=10) as response:
        return json.load(response)


def cached_model_revision(repo_id: str) -> str:
    for repo in scan_cache_dir().repos:
        if repo.repo_id == repo_id:
            revisions = sorted(repo.revisions, key=lambda item: item.last_modified)
            if revisions:
                return revisions[-1].commit_hash
    raise ValueError(f"model is not present in Hugging Face cache: {repo_id}")


def runtime_provenance(subset_dir: Path) -> dict[str, str]:
    inventory_path = subset_dir / "mineru_environment.txt"
    inventory = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    inventory_path.write_text(inventory, encoding="utf-8")
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_revision": git_revision,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "environment_inventory_path": str(inventory_path),
        "environment_inventory_sha256": artifact_sha256(inventory_path),
    }


def execute(
    subset_dir: Path,
    mineru_bin: Path,
    api_url: str,
    smoke: bool,
) -> dict[str, int]:
    health = api_health(api_url)
    if not health.get("protocol_version"):
        raise ValueError(f"invalid MinerU health payload: {health}")
    pages = read_jsonl(subset_dir / "pages.jsonl")
    documents = {row["document_id"]: row for row in read_jsonl(subset_dir / "documents.jsonl")}
    prior_path = subset_dir / "mineru_runs.jsonl"
    prior = {row["page_id"]: row for row in read_jsonl(prior_path)} if prior_path.exists() else {}
    if smoke:
        pages = pages[:1]
    completed = dict(prior)
    runtime = runtime_provenance(subset_dir)
    for page in pages:
        if page["page_id"] in completed and recorded_artifacts_match(completed[page["page_id"]]):
            continue
        document = documents[page["document_id"]]
        output_dir = subset_dir / "mineru" / page["page_id"]
        command = build_mineru_command(
            mineru_bin,
            subset_dir / document["pdf_path"],
            output_dir,
            api_url,
            int(page["source_page_index"]),
        )
        started_at = datetime.now(timezone.utc).isoformat()
        artifacts = run_page(command, output_dir)
        run_id = make_stable_id(
            "run", page["page_id"], "mineru-3.4.0", "hybrid-engine", "high", True
        )
        completed[page["page_id"]] = {
            "run_id": run_id,
            "page_id": page["page_id"],
            "status": "completed",
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "mineru_version": version("mineru"),
            "model_repo": "opendatalab/MinerU2.5-Pro-2605-1.2B",
            "model_revision": cached_model_revision("opendatalab/MinerU2.5-Pro-2605-1.2B"),
            "backend": "hybrid-engine",
            "effort": "high",
            "image_analysis": True,
            "command": command,
            "artifacts": artifacts,
            **runtime,
        }
        write_jsonl(prior_path, [completed[key] for key in sorted(completed)])
    return {"completed_pages": len(completed), "requested_pages": len(pages)}
```

`recorded_artifacts_match()` must recalculate every recorded SHA-256 and return `False` for a missing or changed file. It must not skip based only on page ID.

The CLI must expose:

```text
--subset-dir outputs/sciegqa_mineru_pilot
--mineru-bin $HOME/miniforge3/envs/mineru34/bin/mineru
--api-url http://127.0.0.1:8000
--smoke
```

- [ ] **Step 5: Run focused tests**

Run: `pytest -q tests/test_sciegqa_mineru_runner.py`

Expected: `2 passed`.

- [ ] **Step 6: Start the service and verify health**

In terminal 1:

```bash
MINERU_ENV_PREFIX="$HOME/miniforge3/envs/mineru34" scripts/run_mineru_local_api.sh
```

In terminal 2:

```bash
curl --fail http://127.0.0.1:8000/health | jq .
```

Expected: JSON health payload containing `protocol_version` and task statistics.

- [ ] **Step 7: Run exactly one page through MinerU**

```bash
$HOME/miniforge3/envs/mineru34/bin/python scripts/run_sciegqa_mineru.py \
  --subset-dir outputs/sciegqa_mineru_pilot \
  --mineru-bin "$HOME/miniforge3/envs/mineru34/bin/mineru" \
  --api-url http://127.0.0.1:8000 \
  --smoke
```

Expected: one successful `mineru_runs.jsonl` row and all four required raw artifacts.

- [ ] **Step 8: Inspect the smoke artifacts before continuing**

Run:

```bash
jq . outputs/sciegqa_mineru_pilot/mineru_runs.jsonl
find outputs/sciegqa_mineru_pilot/mineru -type f | sort
```

Expected: backend `hybrid-engine`, effort `high`, image analysis enabled, and hashed content-list, middle, model, and layout artifacts.

- [ ] **Step 9: Run all remaining unique pages**

Run the same command without `--smoke`.

Expected: one successful run row per line in `pages.jsonl`; previously completed page IDs are verified by hash and skipped, not reparsed.

- [ ] **Step 10: Commit the runner after the smoke gate passes**

Do not stage generated outputs. Before committing, run `git branch --show-current`.

```bash
git add scripts/sciegqa_mineru/mineru_runner.py scripts/run_sciegqa_mineru.py tests/test_sciegqa_mineru_runner.py
git commit -m "feat: parse SciEGQA evidence pages with MinerU"
```

## Task 7: Canonical MinerU Segments

**Files:**
- Create: `scripts/sciegqa_mineru/canonicalize.py`
- Create: `tests/test_sciegqa_mineru_canonicalize.py`

- [ ] **Step 1: Write failing canonicalization and IoU tests**

```python
# tests/test_sciegqa_mineru_canonicalize.py
import unittest

from scripts.sciegqa_mineru.canonicalize import canonicalize_content_list, bbox_iou


class CanonicalizeTests(unittest.TestCase):
    def test_preserves_mineru_geometry_and_reading_order(self):
        rows = [
            {"type": "text", "text": "alpha", "bbox": [10, 20, 100, 120], "page_idx": 6},
            {"type": "table", "table_body": "<table></table>", "bbox": [120, 20, 300, 220], "page_idx": 6},
        ]
        segments = canonicalize_content_list("page-1", 6, rows, "run-1")
        self.assertEqual([item["reading_order"] for item in segments], [0, 1])
        self.assertEqual(segments[0]["bbox_norm_1000"], [10.0, 20.0, 100.0, 120.0])
        self.assertEqual(segments[1]["type"], "table")

    def test_iou_is_derived_without_mutating_boxes(self):
        first = [0, 0, 100, 100]
        second = [50, 50, 150, 150]
        self.assertAlmostEqual(bbox_iou(first, second), 2500 / 17500)
        self.assertEqual(first, [0, 0, 100, 100])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `pytest -q tests/test_sciegqa_mineru_canonicalize.py`

Expected: collection ERROR because `canonicalize.py` does not exist.

- [ ] **Step 3: Implement strict canonicalization**

```python
# scripts/sciegqa_mineru/canonicalize.py
from __future__ import annotations

from typing import Any

from scripts.sciegqa_mineru.schema import BBox, make_stable_id


def bbox_iou(first: list[float], second: list[float]) -> float:
    a = BBox.from_list(first)
    b = BBox.from_list(second)
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = (a.x1 - a.x0) * (a.y1 - a.y0)
    area_b = (b.x1 - b.x0) * (b.y1 - b.y0)
    return intersection / (area_a + area_b - intersection) if intersection else 0.0


def canonicalize_content_list(
    page_id: str,
    source_page_index: int,
    rows: list[dict[str, Any]],
    run_id: str,
) -> list[dict[str, Any]]:
    segments = []
    for reading_order, row in enumerate(rows):
        if int(row["page_idx"]) != source_page_index:
            raise ValueError(
                f"MinerU page_idx {row['page_idx']} does not match source index {source_page_index}"
            )
        bbox = BBox.from_list(row["bbox"]).validate((1000, 1000))
        segments.append({
            "segment_id": make_stable_id("segment", run_id, page_id, reading_order, bbox.as_list()),
            "page_id": page_id,
            "run_id": run_id,
            "reading_order": reading_order,
            "type": str(row["type"]),
            "sub_type": row.get("sub_type"),
            "bbox_norm_1000": bbox.as_list(),
            "content": {key: value for key, value in row.items() if key not in {"page_idx", "bbox", "type", "sub_type"}},
        })
    return segments
```

- [ ] **Step 4: Add the real-output canonicalization pass**

Add this function to read every recorded content list, validate joins, and write only derived overlap metadata:

```python
def canonicalize_run_outputs(subset_dir: Path) -> dict[str, int]:
    pages = {row["page_id"]: row for row in read_jsonl(subset_dir / "pages.jsonl")}
    queries = read_jsonl(subset_dir / "queries.jsonl")
    evidence = {row["query_id"]: row for row in read_jsonl(subset_dir / "evidence.jsonl")}
    runs = read_jsonl(subset_dir / "mineru_runs.jsonl")
    segments: list[dict[str, Any]] = []
    for run in runs:
        if run["status"] != "completed":
            raise ValueError(f"run is not complete: {run['run_id']}")
        page = pages[run["page_id"]]
        content_path = Path(run["artifacts"]["content_list"]["path"])
        content_rows = json.loads(content_path.read_text(encoding="utf-8"))
        segments.extend(canonicalize_content_list(
            page_id=page["page_id"],
            source_page_index=int(page["source_page_index"]),
            rows=content_rows,
            run_id=run["run_id"],
        ))
    segment_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        segment_by_page[segment["page_id"]].append(segment)
    overlaps = []
    for query in queries:
        gold = evidence[query["query_id"]]["source_bbox_norm_1000"]
        for segment in segment_by_page[query["page_id"]]:
            iou = bbox_iou(gold, segment["bbox_norm_1000"])
            overlaps.append({
                "query_id": query["query_id"],
                "segment_id": segment["segment_id"],
                "iou": iou,
                "intersects": iou > 0,
            })
    write_jsonl(subset_dir / "segments.jsonl", segments)
    write_jsonl(subset_dir / "query_segment_overlaps.jsonl", overlaps)
    return {"segment_count": len(segments), "overlap_count": len(overlaps)}
```

- [ ] **Step 5: Run focused tests**

Run: `pytest -q tests/test_sciegqa_mineru_canonicalize.py`

Expected: `2 passed`.

- [ ] **Step 6: Canonicalize the smoke page, then all pages**

Run the canonicalization entry point through Task 8's builder first against the smoke output, inspect boxes, then rerun after all page runs complete.

Expected: every segment has a valid page reference, preserved bbox, reading order, and immutable run reference.

- [ ] **Step 7: Commit segment canonicalization**

Before committing, run `git branch --show-current`.

```bash
git add scripts/sciegqa_mineru/canonicalize.py tests/test_sciegqa_mineru_canonicalize.py
git commit -m "feat: canonicalize MinerU page segments"
```

## Task 8: Self-Contained Viewer Bundle

**Files:**
- Create: `scripts/sciegqa_mineru/viewer_bundle.py`
- Create: `scripts/build_sciegqa_mineru_viewer.py`
- Create: `tests/test_sciegqa_mineru_viewer_bundle.py`

- [ ] **Step 1: Write a failing viewer-bundle test**

```python
# tests/test_sciegqa_mineru_viewer_bundle.py
import tempfile
import unittest
from pathlib import Path

from scripts.sciegqa_mineru.viewer_bundle import build_viewer_manifest


class ViewerBundleTests(unittest.TestCase):
    def test_manifest_joins_query_gold_page_and_segments_without_changing_boxes(self):
        manifest = build_viewer_manifest(
            documents=[{"document_id": "doc-1", "arxiv_id": "2401.00001"}],
            pages=[{"page_id": "page-1", "document_id": "doc-1", "image_path": "pages/p.png", "width_px": 2550, "height_px": 3300}],
            queries=[{"query_id": "q-1", "page_id": "page-1", "question": "Q", "answer": "A"}],
            evidence=[{"evidence_id": "e-1", "query_id": "q-1", "page_id": "page-1", "source_bbox_norm_1000": [10, 20, 30, 40]}],
            segments=[{"segment_id": "s-1", "page_id": "page-1", "bbox_norm_1000": [11, 21, 31, 41], "type": "text"}],
            overlaps=[{"query_id": "q-1", "segment_id": "s-1", "iou": 0.8, "intersects": True}],
        )
        self.assertEqual(manifest["queries"][0]["evidence"]["bbox_norm_1000"], [10, 20, 30, 40])
        self.assertEqual(manifest["queries"][0]["segments"][0]["bbox_norm_1000"], [11, 21, 31, 41])
        self.assertEqual(manifest["queries"][0]["segments"][0]["iou"], 0.8)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `pytest -q tests/test_sciegqa_mineru_viewer_bundle.py`

Expected: collection ERROR because `viewer_bundle.py` does not exist.

- [ ] **Step 3: Implement strict joins and self-contained asset copying**

`build_viewer_manifest()` must index each entity by stable ID, reject duplicate/missing references, preserve source arrays exactly, sort queries by approved category order then source row index, and attach only segments from the query's page. `build_site()` must:

```python
def unique_index(rows: list[dict], key: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"duplicate {key}: {value}")
        result[value] = row
    return result


def build_viewer_manifest(documents, pages, queries, evidence, segments, overlaps):
    document_index = unique_index(documents, "document_id")
    page_index = unique_index(pages, "page_id")
    evidence_index = unique_index(evidence, "query_id")
    segments_by_page: dict[str, list[dict]] = defaultdict(list)
    for segment in segments:
        if segment["page_id"] not in page_index:
            raise ValueError(f"segment references missing page: {segment['segment_id']}")
        segments_by_page[segment["page_id"]].append(segment)
    overlap_index = {(row["query_id"], row["segment_id"]): row for row in overlaps}
    category_order = {category: index for index, category in enumerate(CATEGORY_QUOTAS)}
    viewer_queries = []
    for query in sorted(
        queries,
        key=lambda row: (category_order[row["source_category"]], row["source_record_index"]),
    ):
        page = page_index[query["page_id"]]
        document = document_index[page["document_id"]]
        gold = evidence_index[query["query_id"]]
        if gold["page_id"] != page["page_id"]:
            raise ValueError(f"query/evidence page mismatch: {query['query_id']}")
        page_segments = []
        for segment in sorted(segments_by_page[page["page_id"]], key=lambda row: row["reading_order"]):
            overlap = overlap_index[(query["query_id"], segment["segment_id"])]
            page_segments.append(dict(segment, iou=overlap["iou"], intersects=overlap["intersects"]))
        viewer_queries.append({
            **query,
            "document": document,
            "page": page,
            "evidence": {
                **gold,
                "bbox_norm_1000": gold["source_bbox_norm_1000"],
            },
            "segments": page_segments,
        })
    return {"schema_version": "1.0", "queries": viewer_queries}


def build_site(subset_dir: Path, viewer_source_dir: Path, site_dir: Path) -> dict[str, object]:
    site_dir.mkdir(parents=True, exist_ok=True)
    for source in viewer_source_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, site_dir / source.name)
    pages_dir = site_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    documents = read_jsonl(subset_dir / "documents.jsonl")
    pages = read_jsonl(subset_dir / "pages.jsonl")
    queries = read_jsonl(subset_dir / "queries.jsonl")
    evidence = read_jsonl(subset_dir / "evidence.jsonl")
    segments = read_jsonl(subset_dir / "segments.jsonl")
    overlaps = read_jsonl(subset_dir / "query_segment_overlaps.jsonl")
    for page in pages:
        source = subset_dir / page["image_path"]
        destination = pages_dir / f"{page['page_id']}.png"
        shutil.copy2(source, destination)
        page["image_url"] = f"pages/{destination.name}"
    manifest = build_viewer_manifest(documents, pages, queries, evidence, segments, overlaps)
    (site_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "query_count": len(manifest["queries"]),
        "page_asset_count": len(pages),
        "segment_count": len(segments),
    }
```

The source entity JSONL files must remain unchanged. The built site is disposable derived output.

- [ ] **Step 4: Add the build CLI**

`scripts/build_sciegqa_mineru_viewer.py` must canonicalize content lists, write `segments.jsonl` and overlaps, then call `build_site()` with defaults:

```text
--subset-dir outputs/sciegqa_mineru_pilot
--viewer-source-dir viewer/sciegqa_mineru
--site-dir outputs/sciegqa_mineru_pilot/site
```

- [ ] **Step 5: Run focused tests**

Run: `pytest -q tests/test_sciegqa_mineru_viewer_bundle.py tests/test_sciegqa_mineru_canonicalize.py`

Expected: all tests PASS.

- [ ] **Step 6: Commit the bundle builder**

Before committing, run `git branch --show-current`.

```bash
git add scripts/sciegqa_mineru/viewer_bundle.py scripts/build_sciegqa_mineru_viewer.py tests/test_sciegqa_mineru_viewer_bundle.py
git commit -m "feat: build SciEGQA evidence viewer bundle"
```

## Task 9: Page-First Evidence Inspector

**Files:**
- Create: `viewer/sciegqa_mineru/index.html`
- Create: `viewer/sciegqa_mineru/styles.css`
- Create: `viewer/sciegqa_mineru/app.js`
- Create: `tests/test_sciegqa_mineru_viewer_static.py`

- [ ] **Step 1: Write the failing static viewer contract test**

```python
# tests/test_sciegqa_mineru_viewer_static.py
import unittest
from pathlib import Path


class StaticViewerTests(unittest.TestCase):
    def test_viewer_has_required_modes_and_exact_svg_contract(self):
        html = Path("viewer/sciegqa_mineru/index.html").read_text(encoding="utf-8")
        script = Path("viewer/sciegqa_mineru/app.js").read_text(encoding="utf-8")
        styles = Path("viewer/sciegqa_mineru/styles.css").read_text(encoding="utf-8")
        for required_id in ("page-image", "bbox-overlay", "question", "answer", "legend"):
            self.assertIn(f'id="{required_id}"', html)
        for mode in ("gold", "mineru", "overlay"):
            self.assertIn(f'value="{mode}"', html)
        self.assertIn('viewBox="0 0 1000 1000"', html)
        self.assertIn('preserveAspectRatio="none"', html)
        self.assertIn('setAttribute("vector-effect", "non-scaling-stroke")', script)
        self.assertIn("box-gold", styles)
        self.assertIn("box-mineru", styles)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify files are absent**

Run: `pytest -q tests/test_sciegqa_mineru_viewer_static.py`

Expected: FAIL with `FileNotFoundError` for `viewer/sciegqa_mineru/index.html`.

- [ ] **Step 3: Add the semantic page structure**

`index.html` must include these stable IDs used by the application and browser verification:

```html
<main class="inspector">
  <section class="page-pane" aria-label="Evidence page">
    <header class="toolbar">
      <button id="previous-query" type="button">Previous</button>
      <span id="query-position"></span>
      <button id="next-query" type="button">Next</button>
      <button id="zoom-out" type="button">−</button>
      <output id="zoom-value">100%</output>
      <button id="zoom-in" type="button">+</button>
    </header>
    <div id="page-viewport" class="page-viewport">
      <div id="page-stage" class="page-stage">
        <img id="page-image" alt="Scientific evidence page">
        <svg id="bbox-overlay" viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-label="Bounding box overlay"></svg>
      </div>
    </div>
  </section>
  <aside class="details-pane">
    <h1>Evidence inspector</h1>
    <p id="question"></p>
    <p id="answer"></p>
    <fieldset id="layer-mode">
      <legend>Layers</legend>
      <label><input type="radio" name="mode" value="gold"> Gold only</label>
      <label><input type="radio" name="mode" value="mineru"> MinerU only</label>
      <label><input type="radio" name="mode" value="overlay" checked> Overlay</label>
    </fieldset>
    <div id="type-filters"></div>
    <dl id="provenance"></dl>
    <section id="segment-details" aria-live="polite"></section>
    <div id="legend"></div>
  </aside>
</main>
<script src="app.js" defer></script>
```

- [ ] **Step 4: Implement exact SVG geometry and layer behavior**

`app.js` must use source geometry directly:

```javascript
const SVG_NS = "http://www.w3.org/2000/svg";
const state = { manifest: null, index: 0, mode: "overlay", zoom: 1, enabledTypes: new Set() };

function makeRect(box, className, label, payload) {
  const [x0, y0, x1, y1] = box;
  if (!(x0 >= 0 && y0 >= 0 && x1 > x0 && y1 > y0 && x1 <= 1000 && y1 <= 1000)) {
    throw new Error(`Invalid box: ${JSON.stringify(box)}`);
  }
  const rect = document.createElementNS(SVG_NS, "rect");
  rect.setAttribute("x", x0);
  rect.setAttribute("y", y0);
  rect.setAttribute("width", x1 - x0);
  rect.setAttribute("height", y1 - y0);
  rect.setAttribute("class", className);
  rect.setAttribute("vector-effect", "non-scaling-stroke");
  rect.setAttribute("aria-label", label);
  rect.addEventListener("mouseenter", () => showSegment(payload));
  rect.addEventListener("focus", () => showSegment(payload));
  rect.tabIndex = 0;
  return rect;
}

function renderBoxes(query) {
  const overlay = document.querySelector("#bbox-overlay");
  overlay.replaceChildren();
  if (state.mode !== "gold") {
    query.segments
      .filter(segment => state.enabledTypes.has(segment.type))
      .forEach(segment => overlay.append(
        makeRect(segment.bbox_norm_1000, `box box-mineru type-${segment.type}`, `MinerU ${segment.type}`, segment)
      ));
  }
  if (state.mode !== "mineru") {
    // Draw gold last so dense MinerU boxes cannot obscure its solid border.
    overlay.append(makeRect(query.evidence.bbox_norm_1000, "box box-gold", "Gold evidence", query.evidence));
  }
}

function showSegment(payload) {
  const details = document.querySelector("#segment-details");
  details.replaceChildren();
  const title = document.createElement("h2");
  title.textContent = payload.segment_id ? `MinerU ${payload.type}` : "Gold evidence";
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(payload, null, 2);
  details.append(title, pre);
}


function renderLegend(query) {
  const legend = document.querySelector("#legend");
  legend.replaceChildren();
  const gold = document.createElement("p");
  gold.className = "legend-gold";
  gold.textContent = "Solid red — SciEGQA-Train automatic evidence annotation";
  legend.append(gold);
  for (const type of [...new Set(query.segments.map(segment => segment.type))].sort()) {
    const item = document.createElement("p");
    item.className = `legend-mineru type-${type}`;
    item.textContent = `Dashed — MinerU ${type} segment`;
    legend.append(item);
  }
}


function renderFilters(query) {
  const types = [...new Set(query.segments.map(segment => segment.type))].sort();
  state.enabledTypes = new Set(types);
  const container = document.querySelector("#type-filters");
  container.replaceChildren();
  for (const type of types) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = true;
    input.addEventListener("change", () => {
      if (input.checked) state.enabledTypes.add(type); else state.enabledTypes.delete(type);
      renderBoxes(query);
    });
    label.append(input, ` ${type}`);
    container.append(label);
  }
}

function renderQuery() {
  const query = state.manifest.queries[state.index];
  document.querySelector("#query-position").textContent = `${state.index + 1} / ${state.manifest.queries.length}`;
  document.querySelector("#question").textContent = query.question;
  document.querySelector("#answer").textContent = `Answer: ${query.answer}`;
  const image = document.querySelector("#page-image");
  image.src = query.page.image_url;
  image.width = Math.round(900 * state.zoom);
  image.height = Math.round(image.width * query.page.height_px / query.page.width_px);
  document.querySelector("#zoom-value").value = `${Math.round(state.zoom * 100)}%`;
  document.querySelector("#provenance").textContent =
    `${query.document.arxiv_id}${query.document.arxiv_version}, page ${query.page.source_page_number}, ${query.source_category}, ${query.evidence.annotation_origin}`;
  renderFilters(query);
  renderLegend(query);
  renderBoxes(query);
}

function moveQuery(delta) {
  const count = state.manifest.queries.length;
  state.index = (state.index + delta + count) % count;
  renderQuery();
}

async function start() {
  const response = await fetch("manifest.json");
  if (!response.ok) throw new Error(`Manifest request failed: ${response.status}`);
  state.manifest = await response.json();
  document.querySelector("#previous-query").addEventListener("click", () => moveQuery(-1));
  document.querySelector("#next-query").addEventListener("click", () => moveQuery(1));
  document.querySelectorAll('input[name="mode"]').forEach(input => input.addEventListener("change", () => {
    state.mode = input.value;
    renderBoxes(state.manifest.queries[state.index]);
  }));
  document.querySelector("#zoom-in").addEventListener("click", () => { state.zoom = Math.min(2, state.zoom + .25); renderQuery(); });
  document.querySelector("#zoom-out").addEventListener("click", () => { state.zoom = Math.max(.5, state.zoom - .25); renderQuery(); });
  renderQuery();
}

start().catch(error => {
  document.body.textContent = `Viewer failed: ${error.message}`;
});
```

Use DOM `textContent` for all dataset/model text. Never inject untrusted strings with `innerHTML`, derive geometry from image display width, or add manual pixel offsets.

- [ ] **Step 5: Implement the approved responsive visual system**

`styles.css` must preserve the approved page-first split, use red for gold, a type-keyed blue/cyan/purple MinerU palette, dashed MinerU strokes, solid gold strokes, and visible focus states. The page stage must use the image's intrinsic aspect ratio:

```css
.inspector { display: grid; grid-template-columns: minmax(0, 2fr) minmax(20rem, 1fr); min-height: 100vh; }
.page-viewport { overflow: auto; background: #e8edf2; padding: 1rem; }
.page-stage { position: relative; width: max-content; }
.page-stage img { display: block; max-width: none; }
.page-stage svg { position: absolute; inset: 0; width: 100%; height: 100%; }
.box { fill-opacity: .08; stroke-width: 2.5; }
.box-gold { stroke: #dc2626; fill: #ef4444; }
.box-mineru { stroke: #0f766e; fill: #14b8a6; stroke-dasharray: 8 5; }
.type-text { stroke: #2563eb; fill: #3b82f6; }
.type-table { stroke: #0891b2; fill: #06b6d4; }
.type-image, .type-chart { stroke: #7c3aed; fill: #8b5cf6; }
.box:focus { outline: none; stroke-width: 5; }
@media (max-width: 900px) { .inspector { grid-template-columns: 1fr; } .details-pane { order: -1; } }
```

- [ ] **Step 6: Run the static contract test**

Run: `pytest -q tests/test_sciegqa_mineru_viewer_static.py`

Expected: `1 passed`.

- [ ] **Step 7: Build the real viewer bundle**

Run:

```bash
$HOME/miniforge3/envs/mineru34/bin/python scripts/build_sciegqa_mineru_viewer.py \
  --subset-dir outputs/sciegqa_mineru_pilot
```

Expected: `outputs/sciegqa_mineru_pilot/site/manifest.json`, three static source files, and one copied image per unique page.

- [ ] **Step 8: Serve the static site locally**

Run:

```bash
python3 -m http.server 8765 --directory outputs/sciegqa_mineru_pilot/site
```

Expected: `http://127.0.0.1:8765` loads without a server-side application.

- [ ] **Step 9: Preview and iterate before committing styling**

Use the in-app browser to inspect the real page at desktop and narrow widths. Confirm page area, side-panel readability, legend clarity, and dense-box behavior. Apply only targeted CSS/JS fixes revealed by the preview.

- [ ] **Step 10: Commit the approved viewer**

Before committing, run `git branch --show-current`.

```bash
git add viewer/sciegqa_mineru/index.html viewer/sciegqa_mineru/styles.css viewer/sciegqa_mineru/app.js tests/test_sciegqa_mineru_viewer_static.py
git commit -m "feat: add SciEGQA evidence inspection viewer"
```

## Task 10: End-to-End Verification and Operator Documentation

**Files:**
- Modify: `tests/test_sciegqa_mineru_end_to_end.py`
- Create: `docs/mineru_local_setup.md`

- [ ] **Step 1: Add failing final-manifest invariants**

Extend `tests/test_sciegqa_mineru_end_to_end.py` with a test that accepts a built output directory through `SCIEGQA_MINERU_OUTPUT_DIR` and skips only when the variable is absent. When present, assert:

```python
expected_counts = {"q-fin": 4, "q-bio": 3, "eess": 3, "physics": 3, "cs": 2, "econ": 2, "stat": 2, "math": 1}
self.assertEqual(len(queries), 20)
self.assertEqual(len(evidence), 20)
self.assertEqual(Counter(row["source_category"] for row in queries), Counter(expected_counts))
self.assertEqual(len({row["query_id"] for row in queries}), 20)
self.assertEqual({row["page_id"] for row in queries}, {row["page_id"] for row in pages})
self.assertTrue(all(row["annotation_origin"] == "sciegqa_train_automatic" for row in evidence))
self.assertTrue(all(run["status"] == "completed" for run in mineru_runs))
self.assertTrue(segments)
```

- [ ] **Step 2: Run the final artifact test**

Run:

```bash
SCIEGQA_MINERU_OUTPUT_DIR=outputs/sciegqa_mineru_pilot \
pytest -q tests/test_sciegqa_mineru_end_to_end.py
```

Expected: all tests PASS with no skip.

- [ ] **Step 3: Run the complete repository test suite**

Run: `pytest -q`

Expected: all existing and new tests PASS.

- [ ] **Step 4: Verify generated entity counts and references**

Run:

```bash
wc -l outputs/sciegqa_mineru_pilot/{documents,pages,queries,evidence,segments,mineru_runs}.jsonl
jq '.queries | length' outputs/sciegqa_mineru_pilot/site/manifest.json
```

Expected: 20 queries/evidence, 20 viewer queries, one run per unique page, and at least one segment per page.

- [ ] **Step 5: Perform browser interaction verification**

Using the in-app browser at `http://127.0.0.1:8765`, verify:

1. Previous/next visits all 20 query IDs exactly once before wrapping.
2. Gold-only removes every MinerU rectangle.
3. MinerU-only removes the gold rectangle.
4. Overlay displays both sources with different stroke styles and labels.
5. Type filters affect only MinerU rectangles.
6. Hover and keyboard focus show the exact segment ID, type, bbox, content preview, and IoU.
7. Zoom at 50%, 100%, and 200% produces no box drift.
8. Narrow viewport preserves access to page, query, controls, and legend.

- [ ] **Step 6: Compare every unique page to native MinerU visualization**

For each `page_id`, open the rendered source page, the web MinerU-only layer, and its hashed `layout.pdf`. Record pass/fail in `outputs/sciegqa_mineru_pilot/visual_verification.json` with:

```json
{
  "page_id": "page_...",
  "gold_queries_reviewed": ["query_..."],
  "mineru_geometry_matches_native": true,
  "responsive_overlay_has_no_visible_drift": true,
  "observed_parser_errors": []
}
```

Do not edit boxes to make this file pass. A MinerU segmentation mistake belongs in `observed_parser_errors`; a coordinate mismatch fails verification and requires root-cause debugging.

- [ ] **Step 7: Write exact operator documentation**

`docs/mineru_local_setup.md` must contain, in order:

1. machine prerequisites and the 16 GB memory caveat;
2. environment creation;
3. model download/cache location;
4. localhost API launch and health check;
5. subset build;
6. one-page smoke run;
7. full page run;
8. viewer build and static server;
9. verification commands and expected counts;
10. generated artifact layout;
11. license/redistribution warning; and
12. a SOL follow-up note that points to `SOLinstrucitons.md` without modifying the active SOL task.

- [ ] **Step 8: Run verification-before-completion checks**

Invoke `superpowers:verification-before-completion`, rerun the commands it requires, and retain the latest command outputs as evidence. Do not claim success from earlier runs.

- [ ] **Step 9: Request code review**

Invoke `superpowers:requesting-code-review`. Address only findings within this feature's scope, rerun affected tests, and list any observed MinerU parser errors separately from application defects.

- [ ] **Step 10: Commit documentation and final verification test**

Before committing, run `git branch --show-current`.

```bash
git add docs/mineru_local_setup.md tests/test_sciegqa_mineru_end_to_end.py
git commit -m "docs: add MinerU pilot run and verification guide"
```

- [ ] **Step 11: Stop local services cleanly**

Stop the static HTTP server and MinerU API with `Ctrl-C`. Confirm:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8765
```

Expected: both commands fail to connect after shutdown.

## Final Deliverables

- Reproducible MinerU 3.4 local environment and localhost API launcher.
- Deterministic 20-question Train SPSR selection with exact approved quotas.
- Validated arXiv v1 source PDFs and 300-DPI evidence pages under ignored outputs.
- Document/page/query/evidence/segment/run provenance entities with hashes.
- Raw MinerU outputs and native visualization per unique page.
- Page-first static viewer with independent Gold, MinerU, and Overlay modes.
- Automated tests, complete visual-verification record, and operator guide.
