# Retained baseline implementation

Task 6–9 modules, segmentation and Qwen2 integration live here to keep the active
source directory readable. `docprune/__init__.py` explicitly adds this directory
to the package search path. Existing public imports such as `docprune.task9_attribution`
and `docprune.qwen2vl` remain valid. Use those names consistently; `_legacy` is a
physical organization detail, not an alternative public import namespace.

Algorithms and dependencies are unchanged. Regression tests live under
`tests/legacy/`. The owner chose to retain all this code on 2026-09-11.
See the [legacy index](../../../legacy/README.md).
