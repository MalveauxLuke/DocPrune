# Query-Relevant Document Token Pruning

This repository implements and evaluates token compression in document QA.
The goal is to reduce the number of visual/document tokens sent to an
expensive multimodal answerer by pruning content that is irrelevant to the
specific query while preserving all information required to answer correctly.

The central target is:

\[
P(\text{token/region needed for answer}\mid \text{document}, q)
\]

The central research question is where query relevance should be estimated.

The active implementation is a source-grounded, training-free reproduction of
[DocPrune (CVPR 2026)](references/papers/docprune-cvpr-2026.md). It implements
background-, question-, and comprehension-aware pruning for the pinned
Qwen2-VL/M3DocRAG contract. Local equation, layout, adapter, cache, CLI, and
provenance tests pass. No full model, GPU, SOL, accuracy, throughput, memory, or
paper-parity result has been produced yet.

## Start here

- [Agent context](agent-context/INDEX.md)
- [Repository navigation](docs/NAVIGATION.md)
- [Source inventory](docs/SOURCE_INVENTORY.md)
- [Research references](references/README.md)
- [Inherited specifications](docs/specifications/README.md)
- [SOL operating guidance](docs/SOL_INSTRUCTIONS.md)
- [SBATCH examples](examples/sbatch/README.md)
- [DocPrune reproduction guide](docs/reproduction/DOCPRUNE.md)
- [Known reconstruction gaps](docs/reproduction/RECONSTRUCTION_GAPS.md)

Inherited research and historical cluster material are retained for reference.
Only `agent-context/CURRENT_TASK.md` and `sol/CURRENT_SOL_TASK.md` activate work.

## Local verification

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,model]'
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/docprune-m3docvqa inspect \
  --config configs/docprune-m3docvqa.toml --pages 4
```
