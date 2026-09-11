# Source layout

The current `docprune/` package retains shared pruning, retrieval, evaluation and
provenance infrastructure. New corrective-selection training is not implemented.

Older task modules, segmentation and Qwen2 integration live under
[`docprune/_legacy/`](docprune/_legacy/README.md). Their original public import
names remain compatible through the explicit package search path. This organization
does not change their algorithms or declare them compatible with a new reader.

[Legacy index](../legacy/README.md) contains helpers, configs, evidence and tests.
[Current experiment](../docs/experiments/corrective-selection/README.md).
