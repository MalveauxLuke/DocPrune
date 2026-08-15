# SciEGQA-Train MinerU Pilot: Local Runbook

This runbook builds the deterministic 20-question SciEGQA-Train SPSR pilot,
parses each unique gold page with MinerU 3.4, and serves the static evidence
inspector. Generated datasets, PDFs, model outputs, and the built site stay
under the ignored `outputs/` tree.

## 1. Machine prerequisites

- Apple Silicon macOS with Miniforge/Mamba.
- Poppler (`pdftoppm` and `pdfinfo`).
- Git and network access to Hugging Face and arXiv PDF endpoints.
- At least 16 GB unified memory. The local high-effort hybrid backend is
  intentionally single-request/single-window on a 16 GB machine; close other
  CPU/GPU-heavy applications while parsing.

## 2. Create the isolated environment

```bash
mamba env create -f environments/mineru34-macos.yml
$HOME/miniforge3/envs/mineru34/bin/mineru --version
```

Expected MinerU version: `3.4.0` on Python 3.12.

## 3. Model download and cache

The first API start preloads `opendatalab/MinerU2.5-Pro-2605-1.2B` and related
OCR/layout assets from Hugging Face. The primary model is about 2.3 GB. Files
are stored in the normal Hugging Face cache under `~/.cache/huggingface/hub/`.
The run manifest records the exact cached model commit.

## 4. Start the localhost API

```bash
MINERU_ENV_PREFIX="$HOME/miniforge3/envs/mineru34" \
  scripts/run_mineru_local_api.sh
```

The launcher binds only to `127.0.0.1`, permits one concurrent request, and
uses a processing window of one. In another terminal:

```bash
curl --fail http://127.0.0.1:8000/health | python -m json.tool
```

The payload must include `"protocol_version": 2` and `"status": "healthy"`.

## 5. Build the source subset

```bash
$HOME/miniforge3/envs/mineru34/bin/python \
  scripts/build_sciegqa_mineru_subset.py \
  --output-dir outputs/sciegqa_mineru_pilot
```

The builder pins `Yuwh07/SciEGQA-Train` to an immutable Hub commit, selects
exact category quotas, downloads only the selected arXiv v1 PDFs, renders the
annotated pages at 300 DPI, and publishes a complete generation atomically.
It refuses to overwrite an existing output directory.

Current validated generation: 20 queries, 8 documents, and 17 unique pages.
The page count is 17 because one initial source document failed deterministic
dimension validation and was replaced according to the approved selection
rule.

## 6. Run the one-page smoke gate

```bash
$HOME/miniforge3/envs/mineru34/bin/python \
  scripts/run_sciegqa_mineru.py \
  --subset-dir outputs/sciegqa_mineru_pilot \
  --mineru-bin "$HOME/miniforge3/envs/mineru34/bin/mineru" \
  --api-url http://127.0.0.1:8000 \
  --smoke
```

Inspect `mineru_runs.jsonl` and confirm that the run uses `hybrid-engine`,
`high` effort, image analysis, and hashed content-list, middle, model, and
layout artifacts.

## 7. Parse all remaining pages

```bash
$HOME/miniforge3/envs/mineru34/bin/python \
  scripts/run_sciegqa_mineru.py \
  --subset-dir outputs/sciegqa_mineru_pilot \
  --mineru-bin "$HOME/miniforge3/envs/mineru34/bin/mineru" \
  --api-url http://127.0.0.1:8000
```

Completed pages are skipped only after every recorded artifact hash is
recalculated. A changed or incomplete page output fails closed.

## 8. Build and serve the viewer

```bash
$HOME/miniforge3/envs/mineru34/bin/python \
  scripts/build_sciegqa_mineru_viewer.py \
  --subset-dir outputs/sciegqa_mineru_pilot

python3 -m http.server 8765 \
  --bind 127.0.0.1 \
  --directory outputs/sciegqa_mineru_pilot/site
```

Open `http://127.0.0.1:8765`. The layer selector provides Gold only, MinerU
only, and Overlay views. Boxes use the stored normalized coordinates directly;
the viewer performs no resizing correction, clipping, or manual offset.

## 9. Verification

```bash
SCIEGQA_MINERU_OUTPUT_DIR=outputs/sciegqa_mineru_pilot \
  python -m pytest -q tests/test_sciegqa_mineru_end_to_end.py

python -m pytest -q

wc -l outputs/sciegqa_mineru_pilot/{documents,pages,queries,evidence,segments,mineru_runs}.jsonl
```

Expected: 20 query/evidence rows, 17 page/run rows, all eight approved category
counts, a non-empty segment file, 20 viewer queries, and no skipped live-output
test. `visual_verification.json` records page-by-page comparison with MinerU's
native layout PDF; parser mistakes are recorded rather than editing boxes.

## 10. Generated artifact layout

```text
outputs/sciegqa_mineru_pilot/
  dataset_source.json
  selection_audit.json
  documents.jsonl
  pages.jsonl
  queries.jsonl
  evidence.jsonl
  source/<arxiv-id>/...pdf and pages/*.png
  mineru/<page-id>/...content_list.json, middle.json, model.json, layout.pdf
  mineru_runs.jsonl
  segments.jsonl
  query_segment_overlaps.jsonl
  visual_verification.json
  site/index.html, styles.css, app.js, manifest.json, pages/*.png
```

## 11. License and redistribution

The tracked repository contains code and metadata contracts, not redistributed
source PDFs, rendered paper pages, or MinerU model outputs. Review the SciEGQA,
arXiv, MinerU, model, and source-document terms before sharing generated
artifacts. Keep generated material under ignored `outputs/` unless explicit
redistribution permission is established.

## 12. SOL follow-up

Local correctness comes first. Before adapting this workflow to a SOL job,
follow `SOLinstrucitons.md`. When working directly in the SOL checkout, also
read and keep `sol/CURRENT_SOL_TASK.md` current as required by the repository
instructions. This local setup does not modify the active SOL task.
