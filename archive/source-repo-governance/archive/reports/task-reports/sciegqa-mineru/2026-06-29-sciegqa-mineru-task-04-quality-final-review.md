# SciEGQA MinerU Pilot - Task 4 Final Quality Review

## Scope and Verdict

Reviewed multi-dot path fix `0d5c54c` and evidence report `1f11d7b`
against the sole remaining issue in `03dd3c8`.

**Verdict: Approved.** The staged PNG path now exactly models Poppler's literal
`.png` append behavior for both simple and multi-dot prefixes. No production
document or network resource was used.

## Review Evidence

- `scripts/sciegqa_mineru/source_pages.py:178-195` retains the canonical
  destination stem as the Poppler prefix and derives the staged artifact with
  `Path(f"{output_prefix}.png")`. `page.v1.png` therefore resolves to staged
  `page.v1.png`, not `page.png`.
- The change is limited to that path derivation plus one focused regression
  test. Suffix validation, strict PIL verification/full decode, dimension
  validation, hashing, and atomic `os.replace` publication are unchanged.
- The regression test exercises a pre-existing canonical `page.v1.png`, checks
  exact returned dimensions/DPI/SHA-256, and proves no staging artifact remains.
- A real local Poppler probe rendered the same synthetic 72 x 72 point PDF to
  both `page.png` and `page.v1.png`. Each result was 300 x 300 at 300 DPI, and
  each returned hash matched the published file bytes.
- Intentional dimension failures for both names preserved the successful
  canonical bytes exactly and left no hidden temporary directory or file.
- Visual inspection of the multi-dot output showed the complete expected page
  content without clipping or rendering defects.

## Fresh Verification

- `python -m pytest -q tests/test_sciegqa_mineru_source_pages.py`:
  `29 passed, 17 subtests passed in 0.11s`.
- `python -m pytest -q`: `132 passed, 17 subtests passed in 2.09s`.
- `git diff --check 03dd3c8..HEAD`: passed.
- Final temporary-directory contents were exactly `page.png`, `page.v1.png`,
  and `synthetic.pdf`; temporary residue count was zero.

Task 4 source-page reconstruction is approved for the next implementation task.
