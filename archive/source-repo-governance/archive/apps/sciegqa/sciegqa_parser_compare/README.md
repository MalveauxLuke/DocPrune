# SciEGQA parser comparison viewer

This directory contains the static viewer source. Generated review bundles
include page images, `manifest.json`, and linked raw parser outputs, but those
artifacts are not checked into the repository.

Generate a review bundle from a completed parser-comparison run:

```bash
python scripts/build_sciegqa_parser_viewer.py \
  --run-root /path/to/parser/run \
  --viewer-source viewer/sciegqa_parser_compare \
  --site-dir /path/to/parser/run/site
```

Serve the generated `site` directory:

```bash
cd /path/to/parser/run/site
python3 -m http.server 8000
```

Open <http://localhost:8000/> in a browser. Stop the server with `Ctrl-C`.

The viewer supports deep links such as
<http://localhost:8000/?page=0&mode=All%20parsers>.

## Final labeling experiments

Final-labeling modes use DeepSeek-OCR-2 exclusively. Headed pages use complete
Markdown sections; unheaded pages use atomic DeepSeek paragraph blocks.
By default, DeepSeek `image`/`table` runs and their native title blocks form
linked visual bundles from raw grounded reading order. Runs are discovered
top-down, may ignore at most two intervening raw `text` blocks, and search above
before below for one primary title. Unclaimed titles use the preserved
center-distance geometry rule and can merge related runs. The configuration
selector also exposes the original geometry-only strategy for comparison.
Partial overlaps remain marked for manual review rather than becoming binary
labels.

Label colors describe query overlap, not segment identity. Every repeated
`S#` badge marks boxes belonging to one semantic candidate; hover or focus a
box or legend entry to emphasize all linked members. A `document_preamble` is
a full DeepSeek-OCR-2 candidate containing eligible title, author, and
affiliation blocks before the first `##` section.

`MinerU visual links` is a visual-link inspection view. It shows MinerU figures,
charts, tables, and their linked captions with shared `M#` badges and connector
lines; MinerU text is not a final-label source.

Regenerate final-labeling data inside an existing generated bundle:

```bash
python scripts/build_sciegqa_final_labeling.py \
  --manifest /path/to/parser/run/site/manifest.json \
  --output /path/to/parser/run/site/manifest.json
```

Generated bundles are task/run artifacts. Keep them outside tracked source
unless a review explicitly needs a small fixture.
