# SciEGQA MinerU Pilot - Task 4 Quality Re-review

## Scope and Verdict

Re-reviewed source-artifact fix `6bbc16e` and evidence report `0ac7693`
against the findings recorded in `0f91764`. No production document or network
resource was used.

**Verdict: Changes requested.** Both High findings and the geometry,
identifier, payload, and canonical-preservation portions of the Medium findings
are resolved. One valid `.png` path form still fails because the staged output
path does not match Poppler's append behavior.

## Remaining Issue

### Medium - A valid multi-dot `.png` destination resolves to the wrong staged file

`scripts/sciegqa_mineru/source_pages.py:178-184` builds `output_prefix` from
`page_path.stem`, then derives the expected Poppler artifact with
`output_prefix.with_suffix(".png")`. Poppler appends `.png`; `Path.with_suffix`
replaces an existing suffix. For `page.v1.png`, Poppler successfully creates
`page.v1.png` inside the staging directory, but the implementation attempts to
open `page.png` and raises `FileNotFoundError`.

An independent real-Poppler probe reproduced:

```text
FileNotFoundError: .../.page.v1.png.<temporary>/page.png
```

The temporary directory was removed and no canonical file was corrupted, so
the atomicity fix remains sound. However, `page.v1.png` satisfies the new exact
`.png` suffix contract and should not fail. Derive the staged artifact by
appending the renderer suffix, for example `Path(f"{output_prefix}.png")`, and
add a regression test with a multi-dot destination. The test helper at
`tests/test_sciegqa_mineru_source_pages.py:64-66` already models Poppler's
append behavior correctly.

## Resolved Findings

- `scripts/sciegqa_mineru/source_pages.py:128-162` streams PDF bytes to a unique
  sibling temporary file and calls `os.replace` only after header validation,
  SHA-256, and byte sizing. Injected response-read, hash, and replace failures
  preserved the canonical PDF and left only the canonical path.
- Empty and HTML payloads fail before publication; valid `%PDF-` payloads retain
  the exact URL, SHA-256, and byte-count contract.
- `scripts/sciegqa_mineru/source_pages.py:171-195` rejects non-lowercase `.png`
  suffixes before invoking Poppler and uses a same-filesystem temporary
  directory. Subprocess, strict-PIL, dimension, hash, and replace failures all
  preserved canonical page bytes and removed staging artifacts.
- PIL now runs both `verify()` and a reopened full `load()`. A truncated PNG
  whose header still exposed `(20, 30)` was rejected without publication.
- Geometry rejects non-finite and negative coordinates, asymmetric zeros,
  missing axis candidates, inconsistent candidates, and nonpositive resolved
  dimensions. Joint zeros are skipped. The `0.1` boundary is inclusive with
  explicit float tolerance, while `0.100001` fails.
- The known SciEGQA annotation still resolves exactly to `2481 x 3508`.
- URL, metadata-fetch, and download paths share exact unversioned modern arXiv
  ID validation. Versioned, path-bearing, whitespace-padded, and overlong IDs
  fail before network access.
- The renderer remains one page, 300 DPI, `check=True`, with no retry, resize,
  fallback renderer, or new third-party production dependency.
- Response, temporary-file, and PIL handles remain context-managed.

## Verification Evidence

- `python -m pytest -q tests/test_sciegqa_mineru_source_pages.py`:
  `28 passed, 17 subtests passed in 0.11s`.
- `python -m pytest -q`: `131 passed, 17 subtests passed in 2.05s`.
- `git diff --check 0f91764..HEAD`: passed.
- A visually inspected synthetic 72 x 72 point PDF rendered through the real
  Poppler path to an intact 300 x 300 PNG with correct SHA-256 metadata.
- An intentional real-render dimension mismatch preserved the previous PNG and
  left no staging artifact.
- Independent probes covered PDF read, payload, hash, and replace failures;
  page truncation, dimension, hash, replace, and suffix failures; all geometry
  cases listed above; and the exact command/metadata success contracts.

After the staged multi-dot path is corrected and its focused regression passes,
this Task 4 fix is ready for approval re-review.
