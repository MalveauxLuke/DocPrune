# Task 1 Report: Correct processor resources and prove raster mappings

## Implementation

- Added immutable `ColPaliVisualMapping` extraction from the PaliGemma image
  placeholder span. It requires exactly `image_seq_length` placeholders, a
  contiguous unpadded span, a square grid, and unique row-major raster indices.
- Updated processor-contract collection to record visual bounds, raster indices,
  `image_seq_length`, and a schema-v2 ColPali backbone resource pin.
- Made `raster_order_verified` true only when mapping extraction succeeds and
  made contract validation require that check.
- Replaced runnable/test uses of `vidore/colpali-v1` with
  `vidore/colpali-v1.2`; recorded the pinned PaliGemma backbone revision.
- Updated reconstruction-gap handling to describe the verified visual mapping.

## RED evidence

After adding mapping tests, the required focused command failed at collection:

```text
ImportError: cannot import name 'resolve_colpali_visual_mapping'
```

The failure was reproduced with the workspace source explicitly selected:

```bash
PYTHONPATH=src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest \
  tests/test_processor_probe.py tests/test_cli.py -v
```

The configured editable install otherwise resolves to
`/home/lmalveau/DocPrune-runtime-64ea70c`, not this checkout.

## GREEN and verification evidence

```text
PYTHONPATH=src .../python -m pytest tests/test_processor_probe.py tests/test_cli.py -v
17 passed

.../ruff check src/docprune/processor_probe.py tests/test_processor_probe.py tests/test_cli.py
All checks passed!

PYTHONPATH=src .../python -m pytest -v
76 passed
```

All commands used `/home/lmalveau/mamba-envs/docprune-sol/bin/python` (Python
3.10.20). The full suite was run once after the focused checks.

## Files changed

- `src/docprune/processor_probe.py`
- `tests/test_processor_probe.py`
- `tests/test_cli.py`
- `docs/reproduction/RECONSTRUCTION_GAPS.md`

## Self-review

- Reviewed `git diff --check`: no whitespace errors.
- Confirmed `rg` finds no legacy `vidore/colpali-v1` reference in runnable
  Python or tests.
- Confirmed mapping tests cover success plus missing, over-repeated,
  noncontiguous, padded, and non-square placeholder layouts.
- Confirmed no raw IDs or pixels are added to the serialized contract.

## Concerns

- The local `docprune-sol` editable install targets the previous runtime
  checkout. `PYTHONPATH=src` was required so the checks exercised this task's
  source tree; this should be reconciled before an execution handoff uses the
  installed CLI without an explicit source path.

## Review fix round

### Root cause and correction

The initial mapping implementation inferred `range(image_seq_length)` from a
contiguous placeholder span. That proved only sequence shape; a fake processor
could therefore set `raster_order_verified` without establishing the pinned
ColPali/PaliGemma execution contract. It also accepted arbitrary model IDs and
40-character revisions, and the historical handoff snippets still resolved the
legacy moving adapter.

The corrected probe now requires the exact Qwen, ColPali adapter, and PaliGemma
backbone model/revision pairs as explicit CLI and API inputs. It records
auditable ColPali processor evidence and verifies a raster only when all of the
following hold: ColPali Engine 0.3.1, Transformers 4.46.3, the exact
`ColPaliProcessor` class, its `PaliGemmaProcessor` base, its exact
`SiglipImageProcessor`, and the v1.2 1,024-placeholder (32-by-32) visual grid.
The contract records the row-major formula `row * 32 + column`; a fake
processor retains any shape inference but cannot claim raster verification.

Both executable handoffs now provide the immutable v1.2 adapter and backbone
arguments and no longer query the old moving `main` revision. The older
handoff remains explicitly marked superseded.

### Added regression coverage

- Six literal negative resource cases reject wrong Qwen, adapter, and backbone
  model IDs or revisions.
- CLI forwarding requires both backbone arguments.
- A shape-compatible fake processor is asserted to leave
  `raster_order_verified` false and emit unsupported-processor evidence.

### Fix-round verification

```text
PYTHONPATH=src /home/lmalveau/mamba-envs/docprune-sol/bin/python -m pytest \
  tests/test_processor_probe.py tests/test_cli.py -v
23 passed

/home/lmalveau/mamba-envs/docprune-sol/bin/ruff check \
  src/docprune/processor_probe.py src/docprune/cli.py \
  tests/test_processor_probe.py tests/test_cli.py
All checks passed!

rg -n 'vidore/colpali-v1(?!\.2)' --pcre2 sol/handoffs --glob '*.md'
no matches
```
